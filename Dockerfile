FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for whisper.cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libopenblas-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build whisper.cpp
ARG WHISPER_MODEL_NAME=base.en
RUN git clone https://github.com/ggerganov/whisper.cpp.git /root/whisper.cpp && \
    cd /root/whisper.cpp && \
    bash ./models/download-ggml-model.sh ${WHISPER_MODEL_NAME} && \
    cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/whisper-cli /usr/local/bin/whisper-cli && \
    (cp build/bin/libwhisper.so* /usr/local/lib/ 2>/dev/null || cp build/src/libwhisper.so* /usr/local/lib/ 2>/dev/null || true) && \
    ldconfig

# Copy application
COPY app /app/app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Environment defaults
ENV PORT=8012
ENV WHISPER_MODEL=/root/whisper.cpp/models/ggml-base.en.bin
ENV WHISPER_CLI=/usr/local/bin/whisper-cli

EXPOSE ${PORT}

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
