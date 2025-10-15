#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Ensuring dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip >/dev/null
"$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"

export LOCAL_LLM_NATIVE_HELPER="${LOCAL_LLM_NATIVE_HELPER:-/Applications/NativeCaptureHelper.app/Contents/MacOS/NativeCaptureHelper}"

exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"
