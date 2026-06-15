# Pinned by digest to prevent silent upstream drift (see Dockerfile.gpu
# for the incident that motivated this). To refresh deliberately:
#   docker buildx imagetools inspect python:3.11-slim
FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

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
# Pin whisper.cpp to the last-green commit (see Dockerfile.rocm) so the build is
# reproducible and immune to upstream main churn. Bump deliberately.
ARG WHISPER_CPP_REF=84bd03a438454a82150853dce83818013c6609d2
RUN git clone https://github.com/ggerganov/whisper.cpp.git /root/whisper.cpp && \
    cd /root/whisper.cpp && \
    git checkout ${WHISPER_CPP_REF} && \
    bash ./models/download-ggml-model.sh ${WHISPER_MODEL_NAME} && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/whisper-cli /usr/local/bin/whisper-cli && \
    find build -name 'lib*.so*' -exec cp {} /usr/local/lib/ \; && \
    ldconfig

# Copy application
COPY app /app/app
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Environment defaults
ENV PORT=7706
ENV WHISPER_MODEL=/root/whisper.cpp/models/ggml-base.en.bin
ENV WHISPER_CLI=/usr/local/bin/whisper-cli

EXPOSE ${PORT}

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
