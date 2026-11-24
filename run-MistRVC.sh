#!/bin/bash
# Launch MistRVC (Linux/macOS). Requires ./run-MistRVC-installer.sh.
set -euo pipefail

# Не закрывать окно до нажатия Enter — при успехе и при любой ошибке.
wait_for_key() {
    printf '\nPress Enter to close...'
    read -r </dev/tty 2>/dev/null || true
}
trap wait_for_key EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")"

ENV_PYTHON="env/bin/python"

if [ ! -x "$ENV_PYTHON" ]; then
    echo "Error: the Python environment was not found."
    echo "Run './run-MistRVC-installer.sh' first."
    exit 1
fi

check_internet_connection() {
    echo "Checking the internet connection..."
    if curl -s --max-time 5 https://huggingface.co > /dev/null 2>&1; then
        echo "Internet connection is available."
        INTERNET_AVAILABLE=1
    else
        echo "No internet connection detected."
        INTERNET_AVAILABLE=0
    fi
    echo
}

running_interface() {
    echo "Starting the interface..."
    if [ "$INTERNET_AVAILABLE" -eq 1 ]; then
        echo "Running in ONLINE mode..."
        "$ENV_PYTHON" app.py "$@"
    else
        echo "Running in OFFLINE mode..."
        "$ENV_PYTHON" app.py --offline "$@"
    fi
}

check_internet_connection
running_interface "$@"
