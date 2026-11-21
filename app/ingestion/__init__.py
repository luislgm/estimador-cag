"""Ingesta de datos de la empresa: del archivo en bruto al documento limpio y seudonimizado.

    catálogo -> carga -> parseo (+ reparación) -> validación -> deduplicado -> seudonimización

- catalog.py: qué fuentes existen, cuáles se ingieren y por qué (include / review / exclude).
- loaders.py y parsers.py: de los archivos a una lista de Document, el contrato común.
- cleaning.py: reparar, poner en cuarentena o descartar los registros de presupuestos.
- pii.py: seudonimización consistente de datos personales (RGPD) con Presidio.
- orchestrator.py: une las piezas en una ejecución, con su informe.
- architecture.py: ¿CAG, RAG o híbrido? La decisión, a partir del corpus.

Es la preparación de RAG: los documentos que salen de aquí son los que se indexarán.
"""
