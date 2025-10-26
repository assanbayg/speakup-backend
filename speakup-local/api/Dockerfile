# Builder
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg git build-essential pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev \
  libavdevice-dev libavfilter-dev \
  libswscale-dev libswresample-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
  && pip install --default-timeout=240 --no-cache-dir -r requirements.txt

# Runtime 
FROM python:3.11-slim

# Install only runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local /usr/local

COPY main.py /app/main.py

# Set envs
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1
ENV TORCH_DEVICE=cpu

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
