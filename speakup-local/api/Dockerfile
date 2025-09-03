FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --default-timeout=240 --no-cache-dir \
  "uvicorn[standard]==0.30.6" \
  "fastapi==0.115.4" \
  "httpx==0.27.2" \
  "numpy==1.26.4" \
  "soundfile==0.12.1" \
  "pydub==0.25.1" \
  "TTS==0.22.0" \
  "faster-whisper==1.0.1" \
  "python-multipart==0.0.6"
COPY main.py /app/main.py
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
