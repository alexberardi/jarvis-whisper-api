#!/bin/bash

brew install cmake wget

if [ ! -d ~/whisper.cpp ]; then
  git clone https://github.com/ggerganov/whisper.cpp.git ~/whisper.cpp
fi
cd ~/whisper.cpp

pyenv virtualenv 3.11.8 whisper-api-3.11.8
pyenv local whisper-api-3.11.8

pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio
pip install --upgrade setuptools numpy==1.24.3 coremltools ane-transformers openai-whisper torch torchvision torchaudio

if [ ! -f models/ggml-base.en.bin ]; then
  bash ./models/download-ggml-model.sh base.en
fi

./models/generate-coreml-model.sh base.en

# Optional: convert to CoreML (only if needed)
# python3 convert.py --coreml --model base.en

cmake -B build -DWHISPER_COREML=1
cmake --build build --config Release
sudo cp build/bin/whisper-cli /usr/local/bin/whisper-cli
cd -

