# Deployment Guide

This project now has three deployable parts:

- `api`: FastAPI backend with REST, WebSocket, LangGraph, RAG, ChromaDB, and memory storage.
- `web`: React/Vite browser client, built into static files and served by Nginx.
- `redis`: Optional runtime enhancement layer for rate limiting, answer locks,
  WebSocket presence, and lightweight session/report cache.

The old Gradio UI can remain as an internal demo/debug tool, but it is not the recommended public frontend.

## 1. Production Environment Variables

Backend `.env`:

```env
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.5-flash
LLM_TEMPERATURE=0.7

VISION_API_KEY=
VISION_BASE_URL=
VISION_MODEL=qwen-vl-plus

EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3

CHROMA_PERSIST_DIR=./chroma_data
MEMORY_DB_PATH=./memory_data/memory.db
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports

HOST=0.0.0.0
PORT=8000
API_HOST_PORT=8000
WEB_PORT=80
DEBUG=false

# Local development:
CORS_ALLOWED_ORIGINS=http://127.0.0.1:5173,http://localhost:5173,http://localhost

# Production example:
# CORS_ALLOWED_ORIGINS=https://interview.your-domain.com

# Optional public-trial access gate. Set this before sharing a public URL.
APP_ACCESS_TOKEN=demo-2026

# Redis runtime enhancement layer. Keep false for plain local Python runs.
# docker-compose enables Redis for the API container automatically.
REDIS_ENABLED=false
REDIS_URL=redis://localhost:6379/0
RATE_LIMIT_START_PER_MINUTE=3
RATE_LIMIT_ANSWER_PER_MINUTE=20
RATE_LIMIT_RESUME_PER_HOUR=10
REDIS_LOCK_TTL_MS=60000
SESSION_CACHE_TTL_SECONDS=86400
REPORT_CACHE_TTL_SECONDS=604800
WS_PRESENCE_TTL_SECONDS=90

# React/Vite build-time variables for the included web container.
VITE_API_BASE_URL=/api
VITE_ACCESS_CODE=
```

Frontend `web/.env.local` for local development:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api
VITE_ACCESS_CODE=
```

Frontend production variables for a split deployment:

```env
VITE_API_BASE_URL=https://api.your-domain.com/api
VITE_ACCESS_CODE=
```

Frontend production variables for the included Docker Compose deployment:

```env
VITE_API_BASE_URL=/api
VITE_ACCESS_CODE=
```

With `VITE_API_BASE_URL=/api`, the Nginx web container serves React and proxies
`/api`, `/api/ws`, and `/health` to the backend container.

Do not treat `VITE_ACCESS_CODE` as a secret. Browser variables are visible to users. For public trials, it is better to ask users to type the access code in the UI.

## 2. Local Production-Like Run

Build and start both backend and frontend:

```bash
docker compose build
docker compose up -d
```

Initialize the RAG vector store after API keys are configured:

```bash
docker compose run --rm api python -m scripts.init_vector_store --reset
```

Check the backend:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1/health
```

For frontend development without Docker:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## 3. Recommended Portfolio Deployment

You can use either a single VPS or a split deployment.

Single VPS deployment:

- `web` listens on port `80`.
- `web` serves the React static build.
- `web` proxies `/api` and WebSocket traffic to `api:8000`.
- `api` is only bound to `127.0.0.1:${API_HOST_PORT:-8000}` on the server for debugging.

Split deployment:

- Backend: Render, Railway, Fly.io, or a VPS with Docker Compose.
- Frontend: Vercel, Netlify, Cloudflare Pages, or any static hosting.

Production URL shape:

```text
https://interview.your-domain.com  -> web frontend
https://api.your-domain.com        -> FastAPI backend
wss://api.your-domain.com/api/ws/interview/{session_id}
```

Set backend:

```env
CORS_ALLOWED_ORIGINS=https://interview.your-domain.com
APP_ACCESS_TOKEN=your-demo-access-code
DEBUG=false
```

Set frontend:

```env
VITE_API_BASE_URL=https://api.your-domain.com/api
```

## 4. Backend Deploy With Docker

The Dockerfile supports dynamic cloud ports:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

For platforms like Render that inject `PORT`, no extra Dockerfile change is needed.

Important persistent paths:

```text
/app/chroma_data
/app/memory_data
/app/uploads
/app/exports
```

If your platform provides one persistent disk, mount it and point app data there:

```env
CHROMA_PERSIST_DIR=/data/chroma_data
MEMORY_DB_PATH=/data/memory_data/memory.db
UPLOAD_DIR=/data/uploads
EXPORT_DIR=/data/exports
```

## 5. VPS Deployment With Docker Compose

On the server:

```bash
git clone https://github.com/your-name/ai-mock-interview.git
cd ai-mock-interview
cp .env.example .env
```

Edit `.env`, then run:

```bash
docker compose build
docker compose run --rm api python -m scripts.init_vector_store --reset
docker compose up -d
docker compose ps
docker compose logs -f api
```

Open:

```text
http://server-ip
http://server-ip/health
http://127.0.0.1:8000/docs
```

The compose file persists these host directories:

```text
./data
./chroma_data
./memory_data
./redis_data
./uploads
./exports
```

`redis_data` stores Redis append-only persistence for runtime cache data. The
application treats Redis as an enhancement layer, so Redis failures do not erase
the authoritative interview memory or vector store.

The API service is deliberately bound to `127.0.0.1` on the VPS. Public users
should enter through the `web` service, which proxies `/api` and WebSocket
traffic internally.

## 6. VPS Domain and HTTPS

Point your domain to the VPS IP:

```text
interview.your-domain.com -> server public IP
```

For a quick first deploy, you can expose `WEB_PORT=80` and put Caddy on the host:

```caddyfile
interview.your-domain.com {
    reverse_proxy 127.0.0.1:80
}
```

Then set:

```env
CORS_ALLOWED_ORIGINS=https://interview.your-domain.com
VITE_API_BASE_URL=/api
APP_ACCESS_TOKEN=your-demo-access-code
DEBUG=false
```

## 7. Frontend Static Deploy

Build:

```bash
cd web
npm install
npm run build
```

Deploy `web/dist` to your static hosting provider.

Required production variable:

```env
VITE_API_BASE_URL=https://api.your-domain.com/api
```

## 8. Smoke Test Checklist

After deployment:

```text
1. Open https://interview.your-domain.com/health
2. Open https://interview.your-domain.com
3. Enter the trial access code
4. Start an interview from a real JD
5. Test resume upload
6. Answer one question and confirm WebSocket messages work
7. Stop early and confirm a report is generated
8. Restart the backend and confirm Chroma/memory/upload data is still present
```

## 9. Security Notes

- Rotate any API keys that were ever committed or shared.
- Keep `.env` out of Git and Docker images.
- Set `APP_ACCESS_TOKEN` before sharing a public URL.
- Set `DEBUG=false` in production.
- Limit `CORS_ALLOWED_ORIGINS` to the real frontend domain.
- Back up `chroma_data` and `memory_data` before redeploying or moving servers.
