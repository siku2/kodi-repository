set -e

echoerr() {
  echo "$@" 1>&2
}

exit_with_error() {
  echoerr ERROR: "$@"
  exit 1
}

is_python38() {
  local result
  result=$("$1" -c "import sys; print(sys.version_info > (3, 8, 0))" 2>&1)
  [[ "$result" == "True" ]]
  return
}

find_python38() {
  for exec in "python" "python3" "python38" "python3.8"; do
    if is_python38 $exec; then
      echo "$exec"
      return 0
    fi
  done
  return 1
}

PYTHON=$(find_python38 || exit_with_error "need at least python 3.8 to run")

install_dependencies() {
  echo "installing dependencies"

  "$PYTHON" -m pip install -q \
    konfi \
    pygithub \
    requests
}

update_repository() {
  "$PYTHON" tools/update_repository.py
}

install_dependencies
update_repository