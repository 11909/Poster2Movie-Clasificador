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
- **Idea 7b (ensamble):** `clasificador/ensamble.py` alinea las predicciones de val de los
  miembros por id, las promedia (soft voting, pesos iguales), recalibra los 17 umbrales y guarda
  el combinado (contrato 4). Prueba en `tests/test_ensamble.py`; sin `xfail` en `test_idea7b`.
- **Idea 3 (MLP):** `cabezas.py` acepta `activacion: relu|gelu` y `layernorm: true|false` (capa =
  Linear → LayerNorm → activación → Dropout). Se creyó que GELU y la normalización ayudarían a que
  un MLP sobre 512 números se entrene más estable y combine mejor señales débiles (Misterio,
  Fantasía). Sin las claves nuevas la estructura no cambia (los `.pt` viejos cargan). Valores
  inválidos dan error.
- **Idea 4 (Focal Loss):** `perdidas.py` agrega `FocalLoss` (`tipo: focal`, `gamma`, `pos_weight`).
  Se creyó que bajar el peso de los aciertos fáciles enfocaría el entrenamiento en los géneros
  confusos. Decisión: el `pos_weight` de la BCE se conserva por defecto como α por género
  (γ = 0 reproduce exactamente la BCE de la base); `pos_weight: false` lo quita (`B4_focal_g2_sinpw`).
- **Idea 5 (zero-shot):** `zeroshot.py` (`FRASES`, `similitudes`, `calcular`, `--diagnostico`) y la
  variante `zeroshot` en `codificadores.py` (`CodificadorZeroshot`, comparte el CLIP cargado y da lo
  mismo que `calcular`). Se creyó que 17 pistas con significado darían a la cabeza un mejor punto de
  partida para los géneros raros. Ojo: son lineales en la imagen; con cabeza lineal no agregan
  información (ver `docs/PAREJA_B.md`), por eso también `B5_zeroshot_mlp`.
- Nuevo `clasificador/ranking.py`: ordena los experimentos de una idea contra la base con la regla 6.
- Quitados los `xfail` de las ideas 3, 4 y 5; `tests/test_pareja_b.py` cubre las tres, el predictor
  con un modelo imagen + pistas y el ranking.
- **Para integrar en el repo principal:** una cabeza con GELU o LayerNorm necesita llevar
  `construir_cabeza` de `cabezas.py` a `p2m/clasificador.py` (ver `docs/PAREJA_C.md`).
- Pendiente: correr los experimentos de la Pareja B con el dataset real (rejilla B3, B4, B5).
- Pendiente: publicar `v0.1.0` con la base reproducida (debe dar F1 macro 0.537 en test).
