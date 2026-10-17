"""Estimaciones históricas (ficticias) que se inyectan en el prompt como few-shot examples."""

ESTIMATION_EXAMPLES = [
    {
        "meeting_summary": (
            "El cliente, una distribuidora de material de oficina, necesita una plataforma web "
            "de gestión de inventario. Quiere dar de alta productos, controlar el stock en dos "
            "almacenes y recibir alertas cuando un producto baje del mínimo. Habrá tres roles "
            "(administrador, responsable de almacén y consulta) y pide un dashboard con métricas "
            "de rotación de stock. No necesita app móvil."
        ),
        "estimation": """
## Estimación: Plataforma de Gestión de Inventario

### Desglose de tareas:
1. Diseño UI/UX: 40 horas — 2.000 €
2. Backend API (CRUD inventario y almacenes): 60 horas — 3.000 €
3. Autenticación y roles: 20 horas — 1.000 €
4. Dashboard con métricas y alertas de stock: 30 horas — 1.500 €
5. Testing y QA: 25 horas — 1.250 €

**Total estimado: 175 horas — 8.750 € (tarifa 50 €/h)**
**Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
**Duración estimada: 6-8 semanas**
""",
    },
    {
        "meeting_summary": (
            "Una cadena de clínicas de fisioterapia quiere una app móvil (iOS y Android) para que "
            "sus pacientes reserven citas, las paguen por adelantado y reciban recordatorios. "
            "El personal necesita un panel web para gestionar agendas de los fisioterapeutas y "
            "ver las reservas del día. Los pagos deben hacerse con tarjeta a través de Stripe."
        ),
        "estimation": """
## Estimación: App de Reservas para Clínicas de Fisioterapia

### Desglose de tareas:
1. Diseño UI/UX (app y panel web): 35 horas — 1.750 €
2. App móvil multiplataforma (React Native): 80 horas — 4.000 €
3. Backend API (citas, pacientes, agendas): 50 horas — 2.500 €
4. Integración de pagos con Stripe: 20 horas — 1.000 €
5. Notificaciones push y recordatorios: 15 horas — 750 €
6. Panel web de administración: 30 horas — 1.500 €
7. Testing y QA (incluye pruebas en dispositivos): 30 horas — 1.500 €

**Total estimado: 260 horas — 13.000 € (tarifa 50 €/h)**
**Equipo recomendado: 1 desarrollador móvil + 1 desarrollador backend + 1 diseñador UX (part-time)**
**Duración estimada: 8-10 semanas**
""",
    },
    {
        "meeting_summary": (
            "Una empresa de telecomunicaciones quiere un chatbot con IA para su web que responda "
            "dudas frecuentes de clientes (facturas, tarifas, incidencias). Si el bot no sabe "
            "responder, debe crear un ticket en su CRM (HubSpot) y derivar a un agente humano. "
            "Quieren un panel con métricas de conversaciones resueltas y derivadas."
        ),
        "estimation": """
## Estimación: Chatbot de Atención al Cliente con IA

### Desglose de tareas:
1. Análisis y diseño conversacional: 15 horas — 750 €
2. Backend e integración con LLM (base de conocimiento de FAQs): 40 horas — 2.000 €
3. Integración con CRM HubSpot (tickets y derivación): 25 horas — 1.250 €
4. Widget de chat embebible en la web: 20 horas — 1.000 €
5. Panel de métricas de conversaciones: 20 horas — 1.000 €
6. Testing y QA: 15 horas — 750 €

**Total estimado: 135 horas — 6.750 € (tarifa 50 €/h)**
**Equipo recomendado: 1 desarrollador backend con experiencia en IA + 1 desarrollador frontend (part-time)**
**Duración estimada: 4-5 semanas**
""",
    },
]
