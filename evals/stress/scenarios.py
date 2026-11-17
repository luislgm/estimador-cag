"""Escenarios multi-turno para la prueba de carga del CAG.

Tres perfiles de 20 turnos sobre un mismo proyecto, cada uno pensado para forzar un
comportamiento del sistema:

- growing: el proyecto crece; cada turno añade un requisito coherente.
- pivot: en el turno 5 se cambia el stack (de una web en React a una app en Flutter).
- contradiction: el turno 3 fija un presupuesto de 30.000 € y el turno 8 lo cambia a 80.000 €.

Cada turno declara su fact-tracker: el hecho que deberían recordar los turnos posteriores.
Los hechos son nombres propios o cifras que el modelo copia literalmente (Stripe, Auth0,
30.000...), para que MemoryDriftMetric pueda comprobarlos con una búsqueda exacta, sin
embeddings ni LLM como juez. `supersedes` marca un hecho anterior que este turno invalida
(React tras el pivote, 30.000 tras la contradicción): a partir de ahí ya no se exige
recordarlo, y su presencia en la memoria se mide aparte (hecho obsoleto).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioTurn:
    transcript: str
    fact: str | None = None  # lo que deberían recordar los turnos posteriores
    supersedes: str | None = None  # hecho anterior que este turno deja obsoleto


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[ScenarioTurn, ...]

    def facts_before(self, turn_index: int) -> list[str]:
        """Hechos declarados antes del turno `turn_index` (1-based) que siguen vigentes."""
        previous = self.turns[: turn_index - 1]
        superseded = {t.supersedes for t in previous if t.supersedes}
        return [t.fact for t in previous if t.fact and t.fact not in superseded]

    def stale_facts_before(self, turn_index: int) -> list[str]:
        """Hechos que algún turno anterior a `turn_index` ha dejado obsoletos."""
        return [t.supersedes for t in self.turns[: turn_index - 1] if t.supersedes]


def _refinements(*items: tuple[str, str]) -> tuple[ScenarioTurn, ...]:
    return tuple(ScenarioTurn(text, fact) for text, fact in items)


GROWING = Scenario(
    "growing",
    (
        ScenarioTurn(
            "Queremos estimar Nimbus, una plataforma web para que las cadenas de gimnasios "
            "gestionen las reservas de clases de sus socios.",
            "Nimbus",
        ),
        *_refinements(
            ("Añade inicio de sesión de los socios con Auth0.", "Auth0"),
            ("Los pagos de las cuotas se harán con Stripe.", "Stripe"),
            ("Cada cadena de gimnasios tendrá sus datos separados (multi-tenant).", "tenant"),
            ("Recordatorios de clase por SMS con Twilio.", "Twilio"),
            ("Exportar el listado de reservas a CSV.", "CSV"),
            ("Un registro de auditoría de los cambios que hacen los administradores.", "audit"),
            ("Sincronizar las clases con Google Calendar.", "Google Calendar"),
            ("Campañas de email a los socios con Mailchimp.", "Mailchimp"),
            ("Integración con la contabilidad en Holded.", "Holded"),
            ("Informes de ocupación en Power BI.", "Power BI"),
            ("Chat de soporte para los socios con Zendesk.", "Zendesk"),
            ("Notificaciones al equipo de cada gimnasio por Slack.", "Slack"),
            ("Firma de los contratos de los socios con DocuSign.", "DocuSign"),
            ("Monitorización de errores con Sentry.", "Sentry"),
            ("Mapa de los centros con Mapbox.", "Mapbox"),
            ("Acceso al gimnasio con códigos QR en la app.", "QR"),
            ("Recordatorios también por WhatsApp.", "WhatsApp"),
            ("Inicio de sesión de los empleados con Okta.", "Okta"),
            ("Métricas de rendimiento de la plataforma con Datadog.", "Datadog"),
        ),
    ),
)

PIVOT = Scenario(
    "pivot",
    (
        ScenarioTurn(
            "Queremos estimar Orbita, una aplicación web en React para que los técnicos de una "
            "empresa de mantenimiento consulten sus órdenes de trabajo.",
            "Orbita",
        ),
        ScenarioTurn("El frontend será React con TypeScript.", "React"),
        *_refinements(
            ("El backend lo queremos en Django con PostgreSQL.", "Django"),
            ("Los técnicos inician sesión con Auth0.", "Auth0"),
        ),
        ScenarioTurn(
            "Cambio importante: en lugar de una web en React, será una app móvil en Flutter, "
            "porque los técnicos trabajan siempre desde el móvil.",
            "Flutter",
            supersedes="React",
        ),
        *_refinements(
            ("Que la app funcione sin conexión y sincronice al recuperarla.", "conexion"),
            ("Fotos de las averías adjuntas a cada orden.", "foto"),
            ("Firma del cliente al terminar el trabajo con DocuSign.", "DocuSign"),
            ("Notificaciones push con Firebase.", "Firebase"),
            ("Mapa de las órdenes del día con Mapbox.", "Mapbox"),
            ("Integración con el ERP SAP para las piezas.", "SAP"),
            ("Informes de productividad en Power BI.", "Power BI"),
            ("Monitorización de errores con Sentry.", "Sentry"),
            ("Chat entre técnicos y oficina con Slack.", "Slack"),
            ("Lectura de códigos QR de los equipos.", "QR"),
            ("Avisos al cliente por WhatsApp.", "WhatsApp"),
            ("Exportar las órdenes cerradas a CSV.", "CSV"),
            ("Inicio de sesión de la oficina con Okta.", "Okta"),
            ("Métricas de la app con Datadog.", "Datadog"),
            ("Facturación de las órdenes con Holded.", "Holded"),
        ),
    ),
)

CONTRADICTION = Scenario(
    "contradiction",
    (
        ScenarioTurn(
            "Queremos estimar Vega, un portal web B2B para que los bares hagan pedidos a una "
            "distribuidora de bebidas.",
            "Vega",
        ),
        ScenarioTurn("Los bares pagan con Stripe.", "Stripe"),
        ScenarioTurn("El presupuesto máximo es de 30.000 euros.", "30.000"),
        *_refinements(
            ("Integración con el ERP SAP para el stock.", "SAP"),
            ("Avisos de entrega por SMS con Twilio.", "Twilio"),
            ("Informes de ventas en Power BI.", "Power BI"),
            ("Inicio de sesión de los bares con Auth0.", "Auth0"),
        ),
        ScenarioTurn(
            "Corrección: el presupuesto máximo es de 80.000 euros, no de 30.000.",
            "80.000",
            supersedes="30.000",
        ),
        *_refinements(
            ("App móvil para los repartidores en Flutter.", "Flutter"),
            ("Mapa de rutas de reparto con Mapbox.", "Mapbox"),
            ("Firma de la entrega con DocuSign.", "DocuSign"),
            ("Campañas a los bares con Mailchimp.", "Mailchimp"),
            ("Monitorización de errores con Sentry.", "Sentry"),
            ("Soporte a los bares con Zendesk.", "Zendesk"),
            ("Exportar pedidos a CSV.", "CSV"),
            ("Notificaciones push con Firebase.", "Firebase"),
            ("Avisos al comercial por Slack.", "Slack"),
            ("Pedidos también por WhatsApp.", "WhatsApp"),
            ("Contabilidad en Holded.", "Holded"),
            ("Métricas del portal con Datadog.", "Datadog"),
        ),
    ),
)

SCENARIOS: dict[str, Scenario] = {s.name: s for s in (GROWING, PIVOT, CONTRADICTION)}
