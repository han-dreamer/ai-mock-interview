# FastAPI / WebSocket API Contract

This project exposes the LangGraph interview workflow through both REST and
WebSocket APIs. The REST API is useful for debugging, integration tests, and
non-realtime clients. The WebSocket API is used for realtime interview dialogue.

## Session Lifecycle

1. Create a session.
2. Optionally upload a resume before the graph starts.
3. Start the LangGraph workflow.
4. Submit answers until the graph returns a final report.
5. Optionally stop early and generate a partial report.

## REST Endpoints

### Create session

```http
POST /api/interview/start
Content-Type: application/json
```

```json
{
  "jd_text": "Python FastAPI LangGraph AI application developer position",
  "max_follow_ups": 2,
  "mode": "practice",
  "user_id": "local-user"
}
```

Returns a `session_id` and WebSocket URL.

### Create session with resume

```http
POST /api/interview/start-with-resume
Content-Type: multipart/form-data
```

Fields:

- `jd_text`: job description text
- `resume_file`: PDF, PNG, JPG, or JPEG resume
- `mode`: `practice` or `professional`
- `max_follow_ups`: max follow-up count per question
- `user_id`: stable user id for long-term memory

### Upload resume to an existing session

```http
POST /api/interview/session/{session_id}/resume
Content-Type: multipart/form-data
```

The resume must be uploaded before the LangGraph interview starts.

### Start interview graph

```http
POST /api/interview/session/{session_id}/start
```

Returns the first interviewer turn:

```json
{
  "next": {
    "kind": "question",
    "question_index": 1,
    "total_questions": 5,
    "content": "...",
    "skill_tags": ["FastAPI", "LangGraph"],
    "difficulty": "medium"
  }
}
```

### Submit answer

```http
POST /api/interview/session/{session_id}/answer
Content-Type: application/json
```

```json
{
  "content": "My answer..."
}
```

Returns either a follow-up, the next question, or the final report.

### Stop early

```http
POST /api/interview/session/{session_id}/stop
```

Stops the interview and generates a report from the answers collected so far
when enough assessment data exists.

### Inspect state

```http
GET /api/interview/session/{session_id}/state
```

Returns session metadata, whether the graph has started, latest graph state,
current interviewer turn, and final report if available.

### Get report

```http
GET /api/interview/report/{session_id}
```

Returns the final `PracticeReport`, `ProfessionalReport`, or fallback
`InterviewReport`.

## WebSocket Endpoint

```text
ws://localhost:8000/api/ws/interview/{session_id}
```

Client messages:

```json
{"type": "answer", "content": "My answer..."}
{"type": "stop"}
{"type": "ping"}
```

Backward-compatible aliases:

```json
{"type": "start_interview"}
{"type": "end_interview"}
```

Server messages:

- `status`: graph analysis, processing, resume, or heartbeat state
- `question`: a new planned interview question
- `follow_up`: an adaptive follow-up question
- `interview_end`: interview is complete
- `report`: structured final evaluation report
- `error`: recoverable protocol or processing error

## Engineering Notes

- `SessionManager` owns LangGraph execution and exposes a single stateful API
  for both REST and WebSocket routes.
- Each session has a per-session `asyncio.Lock` to avoid duplicate graph starts
  and concurrent answer submission.
- The graph uses LangGraph checkpoint `thread_id=session_id`, so WebSocket
  reconnects can resume from the latest checkpoint.
- REST and WebSocket share the same report lookup behavior:
  `practice_report`, `professional_report`, then `final_report`.

## Gradio Frontend Integration

The Gradio UI now calls the FastAPI REST API through `httpx` instead of invoking
`SessionManager` directly in-process. This keeps the demo frontend aligned with
the deployable backend service.

Start the backend first:

```bash
uvicorn app.main:app --port 8000
```

Then start the Gradio frontend:

```bash
python -m frontend.gradio_app
```

By default, the frontend calls:

```text
http://127.0.0.1:8000/api
```

Override it when needed:

```bash
INTERVIEW_API_BASE_URL=http://127.0.0.1:8000/api python -m frontend.gradio_app
```

Frontend actions map to REST endpoints:

- Start practice interview: `POST /api/interview/start`, then
  `POST /api/interview/session/{session_id}/start`
- Start professional interview with resume:
  `POST /api/interview/start-with-resume`, then
  `POST /api/interview/session/{session_id}/start`
- Submit answer: `POST /api/interview/session/{session_id}/answer`
- Stop early: `POST /api/interview/session/{session_id}/stop`
