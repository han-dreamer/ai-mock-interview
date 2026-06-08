"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.cache.redis_client import close_redis
from app.config import settings
from app.security import access_control_enabled, is_valid_access_token, token_from_request
from app.services.checkpoint import close_checkpointer, init_checkpointer
from app.services.database import close_database, init_database
from app.services.session_manager import reset_session_manager

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_checkpointer()
    await init_database()
    reset_session_manager()
    try:
        yield
    finally:
        await close_checkpointer()
        await close_database()
        await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Mock Interview Agent",
        description="Multi-Agent powered mock interview system",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def require_trial_access(request: Request, call_next):
        if (
            access_control_enabled()
            and request.url.path.startswith("/api/")
            and request.method != "OPTIONS"
            and not is_valid_access_token(token_from_request(request))
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing access code."},
            )
        return await call_next(request)

    cors_origins = settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*", "X-Access-Code", "Authorization"],
    )

    app.include_router(api_router, prefix="/api")

    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "model": settings.llm_model,
            "debug": settings.debug,
            "access_control": access_control_enabled(),
            "redis_enabled": settings.redis_enabled,
        }

    return app


app = create_app()
