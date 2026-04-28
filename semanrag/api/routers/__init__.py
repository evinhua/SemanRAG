"""API routers package – import all routers for inclusion in the FastAPI app."""

from semanrag.api.routers.document_routes import router as document_router
from semanrag.api.routers.query_routes import router as query_router
from semanrag.api.routers.graph_routes import router as graph_router
from semanrag.api.routers.feedback_routes import router as feedback_router
from semanrag.api.routers.admin_routes import router as admin_router
from semanrag.api.routers.ollama_api import router as ollama_router

all_routers = [
    document_router,
    query_router,
    graph_router,
    feedback_router,
    admin_router,
    ollama_router,
]

__all__ = [
    "document_router",
    "query_router",
    "graph_router",
    "feedback_router",
    "admin_router",
    "ollama_router",
    "all_routers",
]
