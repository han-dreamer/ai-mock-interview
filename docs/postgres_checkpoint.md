# PostgreSQL Persistence

This project uses two persistence layers for interview runtime recovery:

- LangGraph checkpoints: stores graph execution state for each `thread_id`.
- Business session metadata: stores lightweight interview metadata needed by the API layer.

Local development and tests still default to in-memory storage. Docker/production enables PostgreSQL for both layers.

## Configuration

```env
CHECKPOINTER_BACKEND=postgres
SESSION_STORE_BACKEND=postgres
POSTGRES_URL=postgresql://ai_mock:ai_mock@postgres:5432/ai_mock_interview
POSTGRES_POOL_MIN_SIZE=1
POSTGRES_POOL_MAX_SIZE=5
```

Supported values:

- `CHECKPOINTER_BACKEND=memory|postgres`
- `SESSION_STORE_BACKEND=memory|postgres`

`docker-compose.yml` includes a PostgreSQL service and sets both backends to `postgres` for the API container.

## Startup Lifecycle

FastAPI initializes persistence during application startup:

1. `app.main.lifespan` calls `init_checkpointer()`.
2. `CHECKPOINTER_BACKEND=memory` creates a LangGraph `MemorySaver`.
3. `CHECKPOINTER_BACKEND=postgres` creates an `AsyncPostgresSaver` and runs `setup()`.
4. `app.main.lifespan` calls `init_database()`.
5. `SESSION_STORE_BACKEND=postgres` opens a psycopg async connection pool and creates business tables.
6. `SessionManager` is reset so newly compiled LangGraph workflows use the initialized checkpointer.
7. On shutdown, PostgreSQL checkpointer connections, business database pool, and Redis are closed.

## Stored Data

LangGraph checkpointing persists graph state, including interrupted/resumable execution for each `thread_id`.

The business session store persists the `interview_sessions` table:

- `session_id`, `user_id`, `mode`
- `jd_text`, `max_follow_ups`, `status`
- `current_question_index`, `follow_up_count`, `graph_started`
- `resume_text`, `resume_parse_result`
- `conversation_history`, `assessments`, `last_state`
- `persisted_assessment_count`, `final_memory_saved`
- `error_message`, `created_at`, `updated_at`, `completed_at`

This lets the API restore `SessionManager` metadata after a process restart and pair it with LangGraph's persisted checkpoint state.

## Runtime Flow

When a session is created, started, answered, stopped, or completed:

1. `SessionManager` updates the in-memory session object.
2. The latest LangGraph state is mirrored into lightweight session metadata.
3. `persist_session()` upserts metadata into `interview_sessions` when `SESSION_STORE_BACKEND=postgres`.
4. Redis cache is still used as an optional acceleration layer, but PostgreSQL is the durable source for restart recovery in Docker/production.

When an API request references a session that is no longer in process memory:

1. REST/WebSocket endpoints call `ensure_session_loaded(session_id)`.
2. `SessionManager` loads the row from `interview_sessions`.
3. It rebuilds the in-memory `_SessionData`.
4. LangGraph can then read the corresponding checkpoint using the same `thread_id=session_id`.

## Local Development

For plain local Python runs without Docker:

```env
CHECKPOINTER_BACKEND=memory
SESSION_STORE_BACKEND=memory
```

For Docker Compose:

```bash
docker compose up -d --build
```

The API waits for PostgreSQL and Redis health checks before starting.

## Current Scope

This phase provides the persistence foundation for restart recovery.

It does not yet implement full user accounts, historical interview list pages, report management UI, or separate normalized report/message tables. Those can be built in the next phase on top of the `user_id` and `interview_sessions` foundation.
