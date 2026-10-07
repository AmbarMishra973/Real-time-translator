"""Routes package for Real-Time Translator API."""
from backend.routes.pipeline_routes import router as pipeline_router
from backend.routes.websocket_routes import router as websocket_router

__all__ = ["pipeline_router", "websocket_router"]
