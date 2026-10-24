import json

import structlog

from app.logging_config import setup_logging


def _emit(capsys, **fields) -> str:
    structlog.get_logger(module="app.test").info("evento", **fields)
    return capsys.readouterr().out


def test_production_logs_are_json(capsys):
    setup_logging("INFO", "production")

    entry = json.loads(_emit(capsys, model="claude", input_tokens=10))

    assert entry["event"] == "evento"
    assert entry["level"] == "info"
    assert entry["module"] == "app.test"
    assert entry["model"] == "claude"
    assert entry["input_tokens"] == 10
    assert "timestamp" in entry


def test_development_logs_are_readable(capsys):
    setup_logging("DEBUG", "development")

    line = _emit(capsys, model="claude", input_tokens=10)

    assert "evento" in line
    assert "model" in line and "claude" in line
    assert not line.lstrip().startswith("{")


def test_log_level_filters_lower_levels(capsys):
    setup_logging("WARNING", "production")

    structlog.get_logger().info("no_debe_salir")
    structlog.get_logger().warning("si_debe_salir")

    out = capsys.readouterr().out
    assert "no_debe_salir" not in out
    assert "si_debe_salir" in out


def test_context_vars_are_added_to_every_log(capsys):
    setup_logging("INFO", "production")
    structlog.contextvars.bind_contextvars(request_id="abc123")

    entry = json.loads(_emit(capsys))

    assert entry["request_id"] == "abc123"


def test_production_logs_include_exceptions(capsys):
    setup_logging("INFO", "production")

    try:
        raise ValueError("fallo")
    except ValueError:
        structlog.get_logger().exception("error")

    entry = json.loads(capsys.readouterr().out)
    assert "ValueError: fallo" in entry["exception"]
