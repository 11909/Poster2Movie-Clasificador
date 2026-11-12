# Cambios

Formato: una sección por release, la más nueva arriba. Cada una dice qué cambió y por qué se
creyó que ayudaría. Los cambios a los contratos del README también se anotan aquí.

## Sin publicar

- Migración del clasificador desde el repo principal (`p2m/clasificador.py`,
  `p2m/evaluar_clasificador.py`) a este repo, con el repo principal como submódulo fijo en
  `f28928d`.
- Entrenamiento por experimentos `.yaml`, evaluación en val con registro, comparación entre
  modelos y preparación de releases.
- **Contrato 4 (nuevo):** modelo combinado para ensamble (7b) y apilamiento (6) en
  `clasificador/combinado.py`. `predictor` y `evaluar` lo cargan; un `.pt` normal se predice
  como un combinado de un miembro (mismas probabilidades que antes).
- **Contrato 3:** cada evaluación escribe su fila en `resultados/<nombre>/registro_<split>.csv`;
  `resultados/registro.csv` se arma con ellas y ya no va en git (los PR en paralelo no chocan).
- Notebooks por idea (ver README) y `00_reproducir_base` (antes `01_`).
- Pendiente: publicar `v0.1.0` con la base reproducida (debe dar F1 macro 0.537 en test).
