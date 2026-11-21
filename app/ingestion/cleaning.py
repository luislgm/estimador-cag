"""Limpieza de los presupuestos: reparar, poner en cuarentena o descartar.

Tres familias de suciedad, tres tratamientos:

- Reparable (normalize_budget): espacios sobrantes, moneda en minúsculas, importes con
  formato español ("12.500,00"), fechas en dd/mm/aaaa. Se corrige sin preguntar, y la
  reparación queda anotada en el registro.
- Cuarentena: falta un dato que alguien puede completar (el nombre del cliente, una fecha
  ilegible). El registro no se ingiere, pero se conserva para que lo revise su responsable.
- Descarte: el registro está mal de raíz (importe negativo, identificador con un formato
  que no existe, moneda desconocida). Se registra el motivo y no se reintenta.

La validación usa Pandera con lazy=True: en vez de pararse en el primer error, recoge todos,
que es lo que dice qué familias de suciedad trae cada ejecución. Después cada fila va a su
cubeta según el tipo de comprobación que ha fallado (QUARANTINE_CHECKS frente al resto).
Por último, de un mismo presupuesto con varias versiones solo se queda la más reciente.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd
import pandera.pandas as pa

BUDGET_ID_RE = r"^BUDGET-\d{4}-\d{4}$"
CLIENT_CODE_RE = r"^CLI-\d{4}$"
CURRENCIES = ["EUR", "USD", "GBP"]
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")

# Comprobaciones que, si fallan, mandan la fila a cuarentena (falta un dato recuperable).
# Cualquier otra que falle la descarta (el dato está mal de raíz).
QUARANTINE_CHECKS = {"not_nullable"}

BUDGET_SCHEMA = pa.DataFrameSchema(
    {
        "budget_id": pa.Column(str, pa.Check.str_matches(BUDGET_ID_RE)),
        "version": pa.Column(int, pa.Check.ge(1)),
        "client_name": pa.Column(str, nullable=False),
        "client_code": pa.Column(str, pa.Check.str_matches(CLIENT_CODE_RE)),
        "date": pa.Column(str, nullable=False),
        "currency": pa.Column(str, pa.Check.isin(CURRENCIES)),
        "total_amount": pa.Column(float, pa.Check.in_range(0, 10_000_000)),
        "total_hours": pa.Column(int, pa.Check.ge(0)),
    },
    strict=False,  # el resto de campos (contacto, partidas, notas) no se valida aquí
    coerce=True,
)


def _amount(value: Any) -> tuple[float | None, bool]:
    """(importe, reparado). Acepta números y textos con formato español: '12.500,00'."""
    if value is None or isinstance(value, int | float):
        return (float(value) if value is not None else None), False
    text = str(value).strip().replace("€", "").replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})*(,\d+)?", text):
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text.replace(",", ".")), True
    except ValueError:
        return None, True


def _date(value: Any) -> tuple[str | None, bool]:
    """(fecha ISO, reparada). None si no se reconoce el formato."""
    if not value:
        return None, False
    for fmt in DATE_FORMATS:
        try:
            parsed: date = datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
        iso = parsed.isoformat()
        return iso, iso != value
    return None, True


def normalize_budget(raw: dict[str, Any]) -> dict[str, Any]:
    """Registro plano y reparado a partir del JSON del presupuesto, con las reparaciones
    aplicadas en `repairs`."""
    repairs = []

    def clean_str(name: str) -> str | None:
        value = raw.get(name)
        if value is None:
            return None
        stripped = str(value).strip()
        if stripped != value:
            repairs.append(f"{name}: espacios")
        return stripped or None

    currency = clean_str("currency")
    if currency and currency != currency.upper():
        repairs.append("currency: mayúsculas")
        currency = currency.upper()
    amount, amount_fixed = _amount(raw.get("total_amount"))
    if amount_fixed:
        repairs.append("total_amount: formato numérico")
    iso_date, date_fixed = _date(raw.get("date"))
    if date_fixed:
        repairs.append("date: formato de fecha")
    items = raw.get("items") or []
    contact = raw.get("contact") or {}
    return {
        "budget_id": clean_str("budget_id"),
        "version": int(raw.get("version") or 1),
        "client_name": clean_str("client_name"),
        "client_code": clean_str("client_code"),
        "date": iso_date,
        "currency": currency,
        "total_amount": amount,
        "total_hours": sum(int(i.get("hours", 0)) for i in items),
        "items": [
            {"concept": i.get("concept", ""), "hours": int(i.get("hours", 0))} for i in items
        ],
        "contact_name": contact.get("name"),
        "contact_email": contact.get("email"),
        "contact_phone": contact.get("phone"),
        "notes": raw.get("notes") or "",
        "repairs": repairs,
    }


@dataclass
class ValidationResult:
    valid: list[dict[str, Any]] = field(default_factory=list)
    quarantined: dict[str, list[str]] = field(default_factory=dict)  # clave -> motivos
    discarded: dict[str, list[str]] = field(default_factory=dict)
    deduplicated: list[str] = field(default_factory=list)  # versiones antiguas retiradas

    @property
    def repaired(self) -> int:
        return sum(bool(r["repairs"]) for r in self.valid)


def _key(record: dict[str, Any]) -> str:
    return f"{record.get('budget_id')}@v{record.get('version')}"


def validate_budgets(records: list[dict[str, Any]]) -> ValidationResult:
    """Valida los registros normalizados, los reparte en cubetas y deduplica versiones."""
    result = ValidationResult()
    if not records:
        return result
    frame = pd.DataFrame(records)
    failures: dict[int, list[tuple[str, str]]] = {}
    try:
        BUDGET_SCHEMA.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as e:
        for case in e.failure_cases.itertuples():
            if pd.isna(case.index):  # pragma: no cover - fallo de esquema, no de fila
                continue
            check = str(case.check).split("(")[0]
            failures.setdefault(int(case.index), []).append((check, str(case.column)))

    for position, record in enumerate(records):
        problems = failures.get(position)
        if not problems:
            result.valid.append(record)
            continue
        reasons = sorted({f"{column}: {check}" for check, column in problems})
        bucket = (
            result.quarantined
            if all(check in QUARANTINE_CHECKS for check, _ in problems)
            else result.discarded
        )
        bucket[_key(record)] = reasons

    latest: dict[str, dict[str, Any]] = {}
    for record in result.valid:
        current = latest.get(record["budget_id"])
        if current is None or record["version"] > current["version"]:
            if current is not None:
                result.deduplicated.append(_key(current))
            latest[record["budget_id"]] = record
        else:
            result.deduplicated.append(_key(record))
    result.valid = list(latest.values())
    return result
