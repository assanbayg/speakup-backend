# SpeakUp Backend

Backend API for SpeakUp: a conversation practice app for children with speech disabilities.

## Requirements

### Deployment Server

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| vCPU | 2 | 4 |
| Storage | 20 GB | 40 GB |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |

> Tested on: Hetzner Cloud CX32 (4 vCPU, 8 GB RAM)

**Why these specs**: XTTS v2 and Whisper models require ~3-4 GB RAM combined. Ollama needs additional memory for the LLM.

### External Services

- **Supabase** (optional): Only needed for account deletion. Create a project at [supabase.com](https://supabase.com) and get the URL + service role key.

## Local Development

```bash
git clone https://github.com/assanbayg/speakup-backend.git
cd speakup-backend

cp .env.example .env

docker compose up --build
```

First startup takes 5-10 minutes (model downloads).

### Without Docker

```bash
cd api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start Ollama separately
ollama serve &
ollama pull qwen2.5:1.5b-instruct-q4_K_M

# Run API
uvicorn main:app --reload --port 8080
```

## Deployment (Hetzner Cloud)

### 1. Server Setup

```bash
# Create server via Hetzner Console or CLI
# Recommended: CX32 (4 vCPU, 8 GB RAM, 80 GB disk)

# SSH into server
ssh root@<your-server-ip>

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker
```

### 2. Deploy Application

```bash
# Clone repository
git clone https://github.com/assanbayg/speakup-backend.git
cd speakup-backend

# Configure environment
cp .env.example .env
nano .env  # Add Supabase credentials if using auth

# Start services
docker compose up -d

# Pull LLM model (run once)
docker exec -it ollama ollama pull qwen2.5:1.5b-instruct-q4_K_M
```

### 3. Firewall Configuration

```bash
ufw allow 22/tcp    # SSH
ufw allow 8080/tcp  # API
ufw enable
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8080/health
# Expected: {"ok":true}

# Test TTS
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Привет!"}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/stt` | Speech-to-text (multipart audio) |
| `POST` | `/tts` | Text-to-speech |
| `POST` | `/chat` | Streaming chat |
| `POST` | `/chat/sync` | Non-streaming chat |
| `GET` | `/speakers` | List available TTS voices |
| `POST` | `/delete-user` | Delete user account (requires Supabase) |

### Example: Speech-to-Text

```bash
curl -X POST http://localhost:8080/stt \
  -F "file=@recording.wav" \
  -F "language=ru"
```

Response:
```json
{
  "text": "Привет как дела",
  "duration": 2.5,
  "language": "ru",
  "metrics": {
    "avg_confidence": 0.72,
    "wpm": 85.5,
    "word_count": 3,
    "clarity_level": "medium"
  }
}
```

### Example: Chat with Metrics

```bash
curl -X POST http://localhost:8080/chat/sync \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Привет!",
    "metrics": {
      "avg_confidence": 0.72,
      "clarity_level": "medium"
    }
  }'
```

## Configuration

All configuration via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://ollama:11434` | Ollama API endpoint |
| `LLM_MODEL` | `qwen2.5:1.5b-instruct-q4_K_M` | Chat model |
| `WHISPER_MODEL` | `qymyz/whisper-tiny-russian-dysarthria` | STT model |
| `XTTS_LANG` | `ru` | Default TTS language |
| `TTS_FORMAT` | `mp3` | Audio output format |
| `TORCH_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_SECRET_KEY` | — | Supabase service role key |
---

**Note**: Supabase handles only account deletion (requires Admin API). All other auth (sign-in, sign-up, password reset) is handled directly by the [mobile app](https://github.com/assanbayg/speakup).