# PostgreSQL Checkpoint Persistence

This project supports two LangGraph checkpointer backends:

- `memory`: default for local development and tests.
- `postgres`: recommended for Docker/production so LangGraph thread checkpoints survive API process restarts.

## Configuration

```env
CHECKPOINTER_BACKEND=postgres
POSTGRES_URL=postgresql://ai_mock:ai_mock@postgres:5432/ai_mock_interview
```

`docker-compose.yml` includes a `postgres` service and sets these values for the API container.

## Lifecycle

FastAPI initializes the configured checkpointer during application startup:

1. `app.main.lifespan` calls `init_checkpointer()`.
2. `CHECKPOINTER_BACKEND=memory` creates a `MemorySaver`.
3. `CHECKPOINTER_BACKEND=postgres` creates an `AsyncPostgresSaver` and runs `setup()`.
4. `SessionManager` compiles both LangGraph workflows with the initialized checkpointer.
5. On shutdown, the Postgres checkpointer connection is closed.

## Scope

PostgreSQL checkpointing persists LangGraph thread state, including graph checkpoints for each `thread_id`.

It does not, by itself, persist the in-memory `_sessions` dictionary used by the current `SessionManager`.
The next persistence step should add business tables for users, interview sessions, messages, reports,
and resume uploads. After that, API restart recovery can load session metadata from PostgreSQL and pair
it with LangGraph checkpoints.

Recommended next tables:

- `users`
- `interview_sessions`
- `interview_messages`
- `interview_reports`
- `resume_uploads`

## Local Development

Without Docker, keep:

```env
CHECKPOINTER_BACKEND=memory
```

With Docker Compose:

```bash
docker compose up -d --build
```

The API waits for both PostgreSQL and Redis health checks before starting.
