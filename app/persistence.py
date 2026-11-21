"""Base de datos de la ingesta (SQLAlchemy 2): seudónimos y trabajos de ingesta.

Dos tablas, estrechas y de pocas escrituras:

- pseudonym_mappings: (tipo, hash) -> seudónimo. El hash es el HMAC del dato con la sal
  secreta: leer la tabla no permite recuperar los datos originales. Borrar una fila es el
  derecho de supresión (art. 17 del RGPD): ese dato ya no se puede relacionar con su
  seudónimo.
- ingestion_jobs: el estado de cada ejecución lanzada desde la API (pending -> running ->
  completed | failed), con su informe o su error.

En producción es Postgres (DATABASE_URL=postgresql+psycopg://...); los tests usan SQLite.
El esquema lo crean las migraciones de Alembic (alembic/), no la aplicación.
"""

import uuid
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import get_settings


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class PseudonymMapping(Base):
    __tablename__ = "pseudonym_mappings"
    __table_args__ = (UniqueConstraint("entity_type", "original_hash", name="uq_mapping"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(64))
    original_hash: Mapped[str] = mapped_column(String(64))
    pseudonym: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    report: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)


@lru_cache
def session_factory() -> sessionmaker | None:
    """Fábrica de sesiones a partir de DATABASE_URL, o None si no hay base de datos."""
    url = get_settings().database_url
    if not url:
        return None
    return sessionmaker(create_engine(url, pool_pre_ping=True), expire_on_commit=False)


class SqlMappingStore:
    """Almacén de seudónimos en la base de datos (el del servicio)."""

    def __init__(self, sessions: sessionmaker):
        self.sessions = sessions

    def _find(self, session, entity_type: str, original_hash: str) -> PseudonymMapping | None:
        query = select(PseudonymMapping).where(
            PseudonymMapping.entity_type == entity_type,
            PseudonymMapping.original_hash == original_hash,
        )
        return session.scalars(query).first()

    def get_or_create(self, entity_type: str, original_hash: str, candidate: str) -> str:
        with self.sessions() as session:
            row = self._find(session, entity_type, original_hash)
            if row:
                return row.pseudonym
            session.add(
                PseudonymMapping(
                    entity_type=entity_type, original_hash=original_hash, pseudonym=candidate
                )
            )
            try:
                session.commit()
            except IntegrityError:  # otra ejecución lo ha creado a la vez: se usa el suyo
                session.rollback()
                return self._find(session, entity_type, original_hash).pseudonym
            return candidate

    def forget(self, entity_type: str, original_hash: str) -> bool:
        with self.sessions() as session:
            row = self._find(session, entity_type, original_hash)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True


class JobRepository:
    def __init__(self, sessions: sessionmaker):
        self.sessions = sessions

    def create(self, source: str) -> str:
        job_id = str(uuid.uuid4())
        with self.sessions() as session:
            session.add(IngestionJob(id=job_id, source=source, status="pending"))
            session.commit()
        return job_id

    def _update(self, job_id: str, **fields: Any) -> None:
        with self.sessions() as session:
            job = session.get(IngestionJob, job_id)
            for name, value in fields.items():
                setattr(job, name, value)
            session.commit()

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status="running", started_at=_now())

    def complete(self, job_id: str, report: dict) -> None:
        self._update(job_id, status="completed", finished_at=_now(), report=report)

    def fail(self, job_id: str, error: str) -> None:
        self._update(job_id, status="failed", finished_at=_now(), error=error)

    def get(self, job_id: str) -> IngestionJob | None:
        with self.sessions() as session:
            return session.get(IngestionJob, job_id)
