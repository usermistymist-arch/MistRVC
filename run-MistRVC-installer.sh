#!/bin/bash
# MistRVC installer for Linux and macOS.
set -euo pipefail

# Не закрывать окно до нажатия Enter — при успехе и при любой ошибке.
wait_for_key() {
    printf '\nPress Enter to close...'
    read -r </dev/tty 2>/dev/null || true
}
trap wait_for_key EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PYTHON_VERSION="3.11"

OS_NAME="$(uname)"
OS_ARCH="$(uname -m)"

# -- PyTorch device ---------------------------------------------------
TORCH_DEVICE="cpu"
if [ "$OS_NAME" = "Linux" ] && command -v nvidia-smi > /dev/null 2>&1; then
    TORCH_DEVICE="cuda"
fi

echo "============================================="
echo "  MistRVC Installer"
echo "  Python: $PYTHON_VERSION | Device: $TORCH_DEVICE"
echo "============================================="
echo

cd "$(dirname "$0")"
PRINCIPAL="$(pwd)"
MINICONDA_DIR="$HOME/miniconda3"
ENV_DIR="$PRINCIPAL/env"

# -- Miniconda installer URL -----------------------------------------
MINICONDA_BASE="https://repo.anaconda.com/miniconda"
MINICONDA_URL=""

case "$OS_NAME" in
    Darwin)
        if [ "$OS_ARCH" = "arm64" ]; then
            MINICONDA_URL="$MINICONDA_BASE/Miniconda3-py311_26.7.1-1-MacOSX-arm64.sh"
        else
            MINICONDA_URL="$MINICONDA_BASE/Miniconda3-py311_26.7.1-1-MacOSX-x86_64.sh"
        fi
        ;;
    Linux)
        if [ "$OS_ARCH" = "x86_64" ]; then
            MINICONDA_URL="$MINICONDA_BASE/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh"
        else
            echo "Error: automatic Miniconda installation is supported only on x86_64 Linux."
            echo "Install Miniconda/conda manually, then run this script again."
            exit 1
        fi
        ;;
    *)
        echo "Error: unsupported operating system: $OS_NAME"
        exit 1
        ;;
esac

install_miniconda() {
    if [ -x "$MINICONDA_DIR/bin/conda" ] || [ -x "$HOME/.conda/bin/conda" ]; then
        if [ -x "$MINICONDA_DIR/bin/conda" ]; then
            CONDA_EXE="$MINICONDA_DIR/bin/conda"
        else
            CONDA_EXE="$HOME/.conda/bin/conda"
        fi
        echo "Conda is already installed ($CONDA_EXE). Skipping."
        return
    fi

    echo "Miniconda was not found. Downloading and installing..."
    curl -fSL -o miniconda.sh "$MINICONDA_URL"
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f miniconda.sh
    CONDA_EXE="$MINICONDA_DIR/bin/conda"
    echo "Miniconda installed."
    echo
}

create_conda_env() {
    local force_flag="${1:-}"
    echo "Creating Python environment (Python=$PYTHON_VERSION) at $ENV_DIR..."
    if [ -n "$force_flag" ]; then
        "$CONDA_EXE" create --yes --force --prefix "$ENV_DIR" "python=$PYTHON_VERSION"
    else
        "$CONDA_EXE" create --yes --prefix "$ENV_DIR" "python=$PYTHON_VERSION"
    fi
}

install_dependencies() {
    echo "Installing dependencies. This may take a while..."
    local env_python="$ENV_DIR/bin/python"

    "$env_python" -m pip install --upgrade pip setuptools

    if [ "$TORCH_DEVICE" = "cuda" ]; then
        echo "Installing PyTorch (CUDA, cu128)..."
        "$env_python" -m pip install torch==2.11 torchaudio==2.11 torchcodec==0.11 \
            --index-url https://download.pytorch.org/whl/cu128
    elif [ "$OS_NAME" = "Linux" ]; then
        echo "Installing PyTorch (CPU)..."
        "$env_python" -m pip install torch==2.11 torchaudio==2.11 torchcodec==0.11 \
            --index-url https://download.pytorch.org/whl/cpu
    else
        echo "Installing PyTorch (CPU/MPS)..."
        "$env_python" -m pip install torch==2.11 torchaudio==2.11 torchcodec==0.11
    fi

    "$env_python" -m pip install -r "$PRINCIPAL/requirements.txt"
    echo "Dependencies installed."
    echo
}

install_ffmpeg() {
    if command -v ffmpeg > /dev/null 2>&1 && command -v ffprobe > /dev/null 2>&1; then
        echo "FFmpeg is already installed on the system. Skipping."
        return
    fi

    if command -v brew > /dev/null 2>&1; then
        echo "Installing FFmpeg with Homebrew..."
        brew install ffmpeg
    elif command -v apt-get > /dev/null 2>&1; then
        echo "Installing FFmpeg with apt..."
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v pacman > /dev/null 2>&1; then
        echo "Installing FFmpeg with pacman..."
        sudo pacman -Syu --noconfirm ffmpeg
    elif command -v dnf > /dev/null 2>&1; then
        echo "Installing FFmpeg with dnf..."
        sudo dnf install -y ffmpeg --allowerasing || ffmpeg_manual_error
    else
        ffmpeg_manual_error
    fi

    if ! command -v ffmpeg > /dev/null 2>&1 || ! command -v ffprobe > /dev/null 2>&1; then
        ffmpeg_manual_error
    fi
}

ffmpeg_manual_error() {
    echo "Error: could not install FFmpeg automatically."
    echo "Install ffmpeg and ffprobe manually (must be on PATH), then run this script again."
    exit 1
}

verify_installation() {
    echo "Checking the installation..."
    "$ENV_DIR/bin/python" -c "import torch, torchaudio, torchcodec, numpy, scipy, sklearn, librosa, soundfile, gradio, faiss, omegaconf, gdown"
    if "$ENV_DIR/bin/python" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "CUDA is available; voice conversion will use the GPU."
    elif [ "$TORCH_DEVICE" = "cuda" ]; then
        echo "WARNING: CUDA is unavailable, although an NVIDIA GPU was detected."
    else
        echo "The application will run without CUDA (CPU/MPS)."
    fi
    echo
}

install_miniconda

if [ -x "$ENV_DIR/bin/python" ] && "$ENV_DIR/bin/python" --version 2>&1 | grep -q "Python $PYTHON_VERSION"; then
    echo "The Python $PYTHON_VERSION environment at $ENV_DIR already exists. Skipping creation."
    echo
else
    if [ -d "$ENV_DIR" ]; then
        echo "Recreating the environment for Python $PYTHON_VERSION..."
        create_conda_env --force
    else
        create_conda_env
    fi
fi

install_dependencies
install_ffmpeg
verify_installation

echo
echo "MistRVC installed successfully!"
echo "Run the application with: ./run-MistRVC.sh"
echo
