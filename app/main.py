"""ASGI module used by uvicorn."""

from app.api import create_app

app = create_app()
