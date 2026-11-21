"""Entorno de Alembic: la URL de DATABASE_URL y los modelos de app/persistence.py."""

from logging.config import fileConfig

from sqlalchemy import create_engine

from alembic import context
from app.config import get_settings
from app.persistence import Base

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def database_url() -> str:
    url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url
    if not url:
        raise RuntimeError("Falta DATABASE_URL para aplicar las migraciones")
    return url


def run_migrations_offline() -> None:
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    with create_engine(database_url()).connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
