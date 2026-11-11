# Cambios

Formato: una sección por release, la más nueva arriba. Cada una dice qué cambió y por qué se
creyó que ayudaría. Los cambios a los contratos del README también se anotan aquí.

## Sin publicar

- Migración del clasificador desde el repo principal (`p2m/clasificador.py`,
  `p2m/evaluar_clasificador.py`) a este repo, con el repo principal como submódulo fijo en
  `f28928d`.
- Entrenamiento por experimentos `.yaml`, evaluación en val con registro, comparación entre
  modelos y preparación de releases.
- Pendiente: publicar `v0.1.0` con la base reproducida (debe dar F1 macro 0.537 en test).
