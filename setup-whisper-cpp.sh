#!/bin/bash

brew install cmake wget

if [ ! -d ~/whisper.cpp ]; then
  git clone https://github.com/ggerganov/whisper.cpp.git ~/whisper.cpp
fi
cd ~/whisper.cpp

# Create venv inside whisper.cpp and install CoreML conversion dependencies
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install coremltools ane-transformers openai-whisper

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

