"""API router registration."""

from fastapi import APIRouter

from app.api.interview_rest import router as interview_rest_router
from app.api.interview_ws import router as interview_ws_router

api_router = APIRouter()
api_router.include_router(interview_rest_router, prefix="/interview", tags=["interview"])
api_router.include_router(interview_ws_router, prefix="/ws", tags=["websocket"])
