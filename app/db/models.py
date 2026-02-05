"""Database models for jarvis-whisper-api."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Setting(Base):
    """Setting model for storing runtime configuration."""

    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), nullable=False, index=True)
    value = Column(Text, nullable=True)  # JSON-encoded
    value_type = Column(String(50), nullable=False, default="string")
    category = Column(String(100), nullable=False, default="general", index=True)
    description = Column(Text, nullable=True)
    requires_reload = Column(Boolean, default=False)
    is_secret = Column(Boolean, default=False)
    env_fallback = Column(String(255), nullable=True)

    # Multi-tenant scoping (all nullable = system default)
    household_id = Column(String(255), nullable=True, index=True)
    node_id = Column(String(255), nullable=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=True, onupdate=func.now())

    __table_args__ = (
        UniqueConstraint(
            "key", "household_id", "node_id", "user_id", name="uq_setting_scope"
        ),
    )

    def __repr__(self) -> str:
        return f"<Setting(key={self.key}, value={self.value})>"
