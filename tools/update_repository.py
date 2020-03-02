import abc
import dataclasses
import functools
import hashlib
import io
import logging
import re
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union, cast, Callable, TypeVar, IO, Type, Tuple
from xml.etree import ElementTree as ET

import konfi
import requests
from github import Github
from github.GitRelease import GitRelease
from github.GitReleaseAsset import GitReleaseAsset
from github.Repository import Repository

logger = logging.getLogger("update_repository")

T = TypeVar("T")


def static_property(func):
    f_name = func.__name__

    @functools.wraps(func)
    def fget(self):
        value = func(self)
        self.__dict__[f_name] = value
        return value

    return property(fget, doc=func.__doc__)


def static_ns_data(func):
    f_name = func.__name__

    @functools.wraps(func)
    def wrapper(self, ctx: "Context", *args, **kwargs):
        ns = self.ns_data(ctx)
        try:
            return ns[f_name]
        except KeyError:
            v = ns[f_name] = func(self, ctx, *args, **kwargs)
            return v

    return wrapper


@konfi.register_converter(Path)
def path_converter(s: str) -> Path:
    return Path(s)


class _PrefixLogger(logging.Logger):
    def __init__(self, prefix: str, log: logging.Logger):
        self.__dict__ = log.__dict__.copy()
        self.__prefix = prefix

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        return super()._log(level, self.__prefix + msg, args, exc_info, extra, stack_info)


ADDON_METADATA_XPATH = r"""./extension[@point="xbmc.addon.metadata"]"""


def _io_copy(reader: IO[T], writer: IO[T]) -> None:
    buf_size = io.DEFAULT_BUFFER_SIZE

    while True:
        buf = reader.read(buf_size)
        if not buf:
            break

        writer.write(buf)


def _copy_file_from_zip(zfp: zipfile.Path, p: Path) -> None:
    with p.open("wb") as writer:
        with zfp.open("r") as reader:
            _io_copy(reader, writer)


@konfi.template()
class BaseAddon(abc.ABC):
    id: str

    def ns_data(self, ctx: "Context") -> Dict[str, Any]:
        return ctx.ns_data(self.id)

    @property
    def logger(self) -> logging.Logger:
        return _PrefixLogger(f"<{self.id}>  ", logger)

    def addon_path(self, ctx: "Context") -> Path:
        p = ctx.config.output_dir / self.id
        if p.exists() and not p.is_dir():
            raise RuntimeError(f"{p} exists but isn't a directory")

        p.mkdir(exist_ok=True)
        return p

    def version_path(self, ctx: "Context", version: str) -> Path:
        return self.addon_path(ctx) / f"{self.id}-{version}.zip"

    def repo_versions(self, ctx: "Context") -> List[str]:
        versions = []
        for p in self.addon_path(ctx).glob(f"{self.id}-*.zip"):
            _, _, v = p.stem.rpartition("-")
            versions.append(v)

        versions.sort()
        return versions

    @abc.abstractmethod
    def remote_versions(self, ctx: "Context") -> List[str]:
        ...

    @abc.abstractmethod
    def download_version(self, ctx: "Context", version: str) -> None:
        ...

    def setup_version(self, ctx: "Context", version: str) -> None:
        log = self.logger

        log.info("setting version %s", version)
        addon_path = zipfile.Path(self.version_path(ctx, version)) / self.id
        addon_xml_path = addon_path / "addon.xml"
        with addon_xml_path.open() as zfp:
            addon_xml = ET.parse(zfp)

        ctx.addon_xmls.append(addon_xml.getroot())

        metadata: Optional[ET.Element] = addon_xml.find(ADDON_METADATA_XPATH)
        if not metadata:
            log.info("no metadata to extract")
            return

        path_in_repo = self.addon_path(ctx)

        if (news := metadata.find("./news")) is not None:
            log.debug("extracting news to changelog")
            news = cast(ET.Element, news)
            changelog_file = path_in_repo / f"changelog-{version}.txt"
            changelog_file.write_text(news.text.strip())
        else:
            log.info("no news to create changelog with")

        if assets := metadata.find("./assets"):
            assets = cast(ET.Element, assets)
            for asset_el in assets:
                asset_el = cast(ET.Element, asset_el)
                p = Path(asset_el.text)
                log.debug("extracting %s", p.name)
                target_path = path_in_repo / p
                target_path.parent.mkdir(parents=True, exist_ok=True)
                _copy_file_from_zip(addon_path / p, target_path)
        else:
            log.info("no assets to unpack")

    def update(self, ctx: "Context") -> None:
        log = self.logger

        log.info("checking versions")

        log.debug("getting versions in repository")
        repo_versions = set(self.repo_versions(ctx))

        log.debug("getting remote versions")
        remote_versions_ordered = self.remote_versions(ctx)
        remote_versions = set(remote_versions_ordered)

        new_versions = remote_versions - repo_versions
        if not new_versions:
            log.info("already up to date")
            return

        log.info("adding %s version(s)", len(new_versions))
        for ver in new_versions:
            log.debug("downloading version %s", ver)
            self.download_version(ctx, ver)

        newest_version = remote_versions_ordered[-1]
        if newest_version in repo_versions:
            log.debug("no new version")
            return

        self.setup_version(ctx, newest_version)


@konfi.template()
class LocalAddon(BaseAddon):
    local: Path

    @static_property
    def addon_xml(self) -> ET.ElementTree:
        path = self.local / "addon.xml"
        return ET.parse(path)

    @static_property
    def _version(self) -> str:
        return self.addon_xml.getroot().attrib["version"]

    def remote_versions(self, ctx: "Context") -> List[str]:
        return [self._version]

    def download_version(self, ctx: "Context", version: str) -> None:
        if version != self._version:
            raise ValueError(f"invalid version: {version!r}")

        download_path = self.version_path(ctx, version)
        if download_path.exists():
            raise ValueError(f"version {version} already downloaded")
        download_path.parent.mkdir(exist_ok=True)

        parent = self.local.parent
        with zipfile.ZipFile(download_path, "w") as zfp:
            for p in self.local.rglob("*"):
                name = str(p.relative_to(parent))
                zfp.write(p, name)


RE_VERSION = re.compile(r"^v?(\d\.\d\.\d(-.+)?)$")


def download_to_file(fp: BinaryIO, url: str) -> None:
    with requests.get(url, stream=True) as resp:
        for chunk in resp.iter_content(chunk_size=None):
            fp.write(chunk)


@konfi.template()
class GitHubAddon(BaseAddon):
    repo: str

    @static_ns_data
    def _get_repo(self, ctx: "Context") -> Repository:
        return ctx.hub.get_repo(self.repo)

    @static_ns_data
    def _releases(self, _) -> Dict[str, GitRelease]:
        # this is only the initial value, the decorator makes sure that this always returns the same value
        return {}

    def remote_versions(self, ctx: "Context") -> List[str]:
        releases = self._releases(ctx)
        versions = []
        for release in self._get_repo(ctx).get_releases():  # type: GitRelease
            match = RE_VERSION.match(release.tag_name)
            if not match:
                raise ValueError(f"couldn't parse version from: {release.tag_name!r}")

            v = match.group(1)
            releases[v] = release
            versions.append(v)

        versions.sort()
        return versions

    def download_version(self, ctx: "Context", version: str) -> None:
        download_path = self.version_path(ctx, version)
        if download_path.exists():
            raise ValueError(f"version {version} already downloaded")
        download_path.parent.mkdir(exist_ok=True)

        expected_asset_name = f"{self.id}-{version}.zip"
        release = self._releases(ctx)[version]
        for asset in release.get_assets():  # type: GitReleaseAsset
            if asset.name == expected_asset_name:
                addon_asset = asset
                break
        else:
            raise LookupError("no asset found")

        with download_path.open("wb") as fp:
            download_to_file(fp, addon_asset.browser_download_url)


@konfi.template()
class Config:
    github_token: str
    output_dir: Path = Path("addons")
    addons: List[Union[LocalAddon, GitHubAddon]]


@dataclasses.dataclass()
class Context:
    config: Config
    hub: Github

    def __post_init__(self) -> None:
        self._namespaces = {}
        self.addon_xmls: List[ET.Element] = []

    def ns_data(self, ns: str) -> Dict[str, Any]:
        try:
            return self._namespaces[ns]
        except KeyError:
            v = self._namespaces[ns] = {}
            return v


def _io_func_writer(writer: Callable[[T], int], *, cls: Type[io.IOBase] = io.BufferedIOBase) -> IO[T]:
    f = cls()
    f.writable = lambda: True
    f.write = writer
    return f


def _get_addons_xml(path: Path) -> Tuple[ET.ElementTree, ET.Element]:
    if path.exists():
        try:
            tree = ET.parse(path)
        except ET.ParseError as e:
            raise RuntimeError("couldn't parse existing addons.xml file") from e
    else:
        tree = ET.ElementTree()

    root: Optional[ET.Element] = tree.getroot()
    if root is None:
        root = ET.Element("addons")
        tree._setroot(root)
    elif root.tag != "addons":
        raise ValueError(f"addons.xml has invalid root. Expected <addons>, got <{root.tag}>")

    return tree, root


def _replace_xml_element(target: ET.Element, replace_with: ET.Element) -> None:
    target.clear()
    for attr in ("tag", "text", "tail", "attrib"):
        setattr(target, attr, getattr(replace_with, attr))

    for child in replace_with:
        target.append(child)


def update_addons_xml(ctx: Context) -> None:
    addon_xmls = ctx.addon_xmls
    if not addon_xmls:
        logger.debug("no need to update addons.xml")
        return

    logger.info("updating addons.xml")
    path = ctx.config.output_dir / "addons.xml"
    tree, addons_el = _get_addons_xml(path)

    for addon_xml in addon_xmls:
        addon_id: str = addon_xml.attrib["id"]

        if (existing := addons_el.find(f"./addon[@id=\"{addon_id}\"]")) is not None:
            existing = cast(ET.Element, existing)
            logger.debug("replacing existing addon data for: %s", addon_id)
            _replace_xml_element(existing, addon_xml)
        else:
            addons_el.append(addon_xml)

    hasher = hashlib.md5()
    with path.open("wb") as fp:
        def write(s: bytes) -> int:
            hasher.update(s)
            return fp.write(s)

        tree.write(_io_func_writer(write))

    logger.debug("updating addons.xml.md5")
    path.with_suffix(".xml.md5").write_text(hasher.hexdigest())


def load_config() -> Config:
    konfi.set_sources(konfi.FileLoader("config.yaml"), konfi.Env(decoder="yaml"))
    return konfi.load(Config)


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("{name:^25} | {levelname:>6}: {message}", style="{"))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def main():
    config = load_config()
    ctx = Context(config, Github(config.github_token))

    config.output_dir.mkdir(exist_ok=True)

    logger.info("updating %s addons", len(config.addons))
    handles = []
    for addon in config.addons:
        th = threading.Thread(target=addon.update, args=(ctx,))
        th.start()
        handles.append(th)

    logger.debug("waiting for all addons to complete")
    for th in handles:
        th.join()

    update_addons_xml(ctx)


if __name__ == "__main__":
    setup_logging()
    main()
