"""Shared SQLAlchemy connection helpers for the snapshot pipeline."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def create_sqlite_engine(db_path: str) -> Engine:
    """Create an Engine for an existing SQLite database path."""
    absolute_path = Path(db_path).resolve()
    return create_engine(f"sqlite:///{absolute_path}")
