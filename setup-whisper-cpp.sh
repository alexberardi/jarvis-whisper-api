#!/bin/bash
set -euo pipefail

OS="$(uname -s)"
if [ "$OS" != "Linux" ]; then
  echo "This setup script targets Linux/Ubuntu. Detected: $OS"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  wget \
  pkg-config \
  libopenblas-dev \
  libsndfile1 \
  ffmpeg

if [ ! -d "$HOME/whisper.cpp" ]; then
  git clone https://github.com/ggerganov/whisper.cpp.git "$HOME/whisper.cpp"
fi
cd "$HOME/whisper.cpp"

MODEL_NAME="${WHISPER_MODEL_NAME:-base.en}"
MODEL_FILE="models/ggml-${MODEL_NAME}.bin"
if [ ! -f "$MODEL_FILE" ]; then
  bash ./models/download-ggml-model.sh "$MODEL_NAME"
fi

CUDA_FLAG=""
if [ "${WHISPER_ENABLE_CUDA:-false}" = "true" ] || [ "${WHISPER_ENABLE_CUDA:-0}" = "1" ]; then
  sudo apt-get install -y nvidia-cuda-toolkit
  CUDA_FLAG="-DGGML_CUDA=ON"
fi

cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS ${CUDA_FLAG}
cmake --build build --config Release -j

if [ -f build/bin/whisper-cli ]; then
  sudo cp build/bin/whisper-cli /usr/local/bin/whisper-cli
fi

LIB_DIR=""
if [ -f build/bin/libwhisper.so ]; then
  LIB_DIR="build/bin"
elif [ -f build/src/libwhisper.so ]; then
  LIB_DIR="build/src"
fi

if [ -n "$LIB_DIR" ]; then
  sudo cp "$LIB_DIR"/libwhisper.so* /usr/local/lib/ 2>/dev/null || true
  sudo ldconfig || true
fi

cd -

