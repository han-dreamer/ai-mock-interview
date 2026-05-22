# AI Mock Interview Web

This is the production-oriented web client for the AI Mock Interview project.
The original Gradio UI remains useful for internal demos and quick debugging;
this React/Vite app is intended for portfolio presentation and real trial users.

## Local Development

Start the FastAPI backend first:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then start the web app:

```bash
cd web
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Environment Variables

Create `web/.env.local` when the backend is not running at the default URL:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api
VITE_ACCESS_CODE=
```

For production:

```env
VITE_API_BASE_URL=https://api.your-domain.com/api
VITE_ACCESS_CODE=
```

For the repository's Docker Compose deployment, use a same-origin API path:

```env
VITE_API_BASE_URL=/api
VITE_ACCESS_CODE=
```

The included Nginx config serves the built React app and proxies `/api`,
`/api/ws`, and `/health` to the FastAPI container.

The client derives the WebSocket URL from `VITE_API_BASE_URL`, so HTTPS
automatically becomes WSS.

If the backend sets `APP_ACCESS_TOKEN`, users must enter the same value in the
trial access code field before starting an interview. `VITE_ACCESS_CODE` can
pre-fill that field for private demos, but it is not a secret once shipped to a
public browser.

## Build

```bash
npm run build
```

The static build output is written to `web/dist`.
