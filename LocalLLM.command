#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Prefer a modern Python (3.12/3.11/3.10), but allow override via $PYTHON_BIN
pick_python() {
  if [ -n "${PYTHON_BIN:-}" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "$PYTHON_BIN"; return 0
  fi
  candidates=(
    /opt/homebrew/bin/python3.12
    /opt/homebrew/bin/python3.11
    /opt/homebrew/bin/python3.10
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
    python3.12 python3.11 python3.10 python3
  )
  for c in "${candidates[@]}"; do
    if command -v "$c" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  echo python3
}

PYTHON_BIN="$(pick_python)"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# --- Recreate venv if interpreter major.minor changed
want_ver="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
have_ver=""
if [ -x "$VENV_PYTHON" ]; then
  have_ver="$($VENV_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
fi

if [ ! -x "$VENV_PYTHON" ] || [ "$want_ver" != "$have_ver" ]; then
  if [ -d "$VENV_DIR" ]; then
    echo "Recreating virtual environment with Python $want_ver (was ${have_ver:-none})..."
    rm -rf "$VENV_DIR"
  else
    echo "Creating virtual environment with Python $want_ver..."
  fi
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Ensuring dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip >/dev/null
"$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"

# Help Tk on macOS avoid deprecation panics with old system frameworks.
export TK_SILENCE_DEPRECATION=1

export LOCAL_LLM_NATIVE_HELPER="${LOCAL_LLM_NATIVE_HELPER:-/Applications/NativeCaptureHelper.app/Contents/MacOS/NativeCaptureHelper}"

exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"
