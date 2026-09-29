"""WSGI entry point for a production server such as Gunicorn."""

from app import app

__all__ = ["app"]
