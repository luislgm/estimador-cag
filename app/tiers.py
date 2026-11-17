"""Tier de audiencia: para quién es la estimación, decidido en cada turno.

El tier ajusta el enfoque de la respuesta (no su formato, que siguen decidiendo
output_format y detail_level):

- executive: dirección. Coste total, plazo, decisión y riesgos de negocio, sin jerga.
- pm: gestión del proyecto. Fases, hitos, dependencias y equipo por rol.
- developer: equipo técnico. Arquitectura, tecnologías concretas y riesgos técnicos.
- default: sin enfoque especial (el prompt no cambia).

Cómo se decide, por orden:

1. Override: si la petición trae un tier, manda siempre.
2. Reglas: una cadena de expresiones regulares sobre el mensaje del usuario, por orden de
   prioridad; gana la primera que coincide. Primero las menciones explícitas a la audiencia
   ("para el comité de dirección"), después las señales del tema (retorno de la inversión,
   hitos, Kubernetes...).
3. Herencia: si el mensaje no da pistas, se mantiene el tier del turno anterior, para que no
   cambie de un turno a otro sin motivo ("quita la descarga de facturas" no dice nada de la
   audiencia, pero sigue siendo para la misma persona).
4. Si no hay nada de lo anterior: default.

Sin llamadas al LLM: cuesta cero, es determinista y explicable. La resolución devuelve qué
regla decidió el tier, para mostrarlo ("executive · regla audiencia_direccion").

Solo se mira el mensaje del usuario, no los adjuntos ni la memoria: un documento técnico
adjunto no convierte en técnica una conversación con dirección, y la memoria siempre tiene
tecnologías (las rellena el propio modelo), así que la regla técnica saltaría siempre.
"""

import re
from dataclasses import dataclass
from enum import StrEnum


class Tier(StrEnum):
    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


TIER_LABELS = {
    Tier.EXECUTIVE: "Dirección",
    Tier.PM: "Gestión del proyecto (PM)",
    Tier.DEVELOPER: "Equipo técnico",
    Tier.DEFAULT: "General",
}

OVERRIDE_RULE = "override"
DEFAULT_RULE = "sin_señales"


@dataclass(frozen=True)
class TierResolution:
    tier: Tier
    rule: str  # la regla que lo decidió: override, una de TIER_RULES o sin_señales
    inherited: bool = False  # True si se mantiene del turno anterior


# Reglas por orden de prioridad: (nombre, tier, patrón)
TIER_RULES: tuple[tuple[str, Tier, re.Pattern[str]], ...] = (
    # 1. Menciones explícitas a quien lo va a leer
    (
        "audiencia_direccion",
        Tier.EXECUTIVE,
        re.compile(
            r"\b(comit[eé] de direcci[oó]n|direcci[oó]n general|consejo de administraci[oó]n"
            r"|junta directiva|ceo|cfo|inversores|board|executive summary|resumen ejecutivo"
            r"|para (la|el) (direcci[oó]n|gerencia|gerente|director[a]?))\b",
            re.I,
        ),
    ),
    (
        "audiencia_tecnica",
        Tier.DEVELOPER,
        re.compile(
            r"\b(equipo t[eé]cnico|desarrolladores|developers|cto|arquitecto de software"
            r"|para (los|el) (devs?|programadores|t[eé]cnicos))\b",
            re.I,
        ),
    ),
    (
        "audiencia_pm",
        Tier.PM,
        re.compile(
            r"\b(project manager|jefe de proyecto|jefa de proyecto|pmo|product owner"
            r"|scrum master)\b",
            re.I,
        ),
    ),
    # 2. Señales del tema
    (
        "negocio",
        Tier.EXECUTIVE,
        re.compile(
            r"\b(retorno de la inversi[oó]n|roi|business case|caso de negocio|rentabilidad"
            r"|aprobar el presupuesto|go/no[- ]go)\b",
            re.I,
        ),
    ),
    (
        "planificacion",
        Tier.PM,
        re.compile(
            r"\b(hitos?|milestones?|roadmap|cronograma|diagrama de gantt|gantt|sprints?"
            r"|entregables|planificaci[oó]n|calendario de entregas|dependencias entre fases)\b",
            re.I,
        ),
    ),
    (
        "detalle_tecnico",
        Tier.DEVELOPER,
        re.compile(
            r"\b(docker|kubernetes|k8s|microservicios?|ci/cd|terraform|arquitectura"
            r"|api rest|graphql|endpoints?|kafka|esquema de (la )?base de datos|serverless"
            r"|stack t[eé]cnico)\b",
            re.I,
        ),
    ),
)


def resolve_tier(
    message: str,
    previous: TierResolution | None = None,
    override: Tier | None = None,
) -> TierResolution:
    """El tier de este turno a partir del mensaje del usuario y del tier anterior."""
    if override is not None:
        return TierResolution(override, OVERRIDE_RULE)
    for name, tier, pattern in TIER_RULES:
        if pattern.search(message):
            return TierResolution(tier, name)
    if previous is not None and previous.tier != Tier.DEFAULT:
        return TierResolution(previous.tier, previous.rule, inherited=True)
    return TierResolution(Tier.DEFAULT, DEFAULT_RULE)
