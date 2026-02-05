"""Database session configuration for jarvis-whisper-api."""

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@lru_cache
def get_engine():
    """Get the SQLAlchemy engine."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@host.docker.internal:5432/jarvis_whisper"
    )
    return create_engine(database_url)


@lru_cache
def get_session_local():
    """Get the SessionLocal class for creating database sessions."""
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
