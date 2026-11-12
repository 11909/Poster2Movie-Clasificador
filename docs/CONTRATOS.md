# Contratos

[← README](../README.md)


Formatos fijos para que las tres parejas trabajen en paralelo. Cambiarlos rompe el trabajo de
otra pareja: solo cambian por acuerdo del equipo y se anotan en `CHANGELOG.md`.

### Contrato 1 — Vectores (los produce A; los usan B y C)

Un archivo por codificador y variante, en `<P2M_SALIDAS>/vectores/`:

```
<codificador>__<variante>.npz     ids: int64 (N,) id de TMDB · vectores: float32 (N, D)
<codificador>__<variante>.json    nombre, codificador, variante, modelo_hf, dim, peliculas, fecha
```

- Nombres: `clip_b32__centro`, `clip_l14__centro`, `siglip_base__3recortes`, `clip_b32__ocr`,
  `clip_b32__zeroshot`... Doble guion bajo entre codificador y variante.
- `clip_b32__centro` ya existe: son los `embeddings_clip_b32.npz` del dataset compartido.
- Se generan con `python -m clasificador.codificadores <codificador> --variante <variante>`.
- La **misma clase** que genera el archivo es la que usa `predictor.py`, para que el cartel se
  procese igual al entrenar y al predecir. Toda clase nueva hereda de `Codificador`
  (`nombre`, `modelo_hf`, `dim`, `codificar(imagenes) -> ndarray`) y se registra en
  `crear_codificador`.
- Las pistas zero-shot y el texto OCR también son archivos de este contrato (vectores de N × 17
  o N × 512), así se combinan con cualquier otro.

### Contrato 2 — Experimento (lo usan todos)

Un `.yaml` en `experimentos/`:

```yaml
nombre: B_focal_g2      # = nombre del archivo
pareja: B               # A, B, C o base
descripcion: Focal Loss con gamma 2 sobre la base.
vectores:               # uno o varios del contrato 1; varios se concatenan por id
  - clip_b32__centro
cabeza: {tipo: lineal}  # lineal | mlp (ocultas, dropout) | ...
perdida: {tipo: focal, gamma: 2}
entrenamiento: {epocas: 200, tasa: 0.001, paciencia: 10, lote: 256, weight_decay: 0.0001, semilla: 42}
```

`python -m clasificador.entrenar experimentos/<nombre>.yaml` produce:

- `<P2M_SALIDAS>/modelos/<nombre>.pt` con las claves del `.pt` del repo principal
  (`modelo_clip`, `embeddings`, `arquitectura`, `generos`, `capa`, `umbrales`) más `vectores`,
  `experimento` y `commit`.
- `<P2M_SALIDAS>/predicciones/<nombre>.npz` con `ids_val` y `prob_val` (N × 17): lo que usan el
  ensamble y el apilamiento.

Un tipo nuevo de cabeza o de pérdida se agrega en `cabezas.py` o `perdidas.py` y se documenta en este archivo.

### Contrato 3 — Métricas (lo usan todos)

`python -m clasificador.evaluar <nombre>` escribe en `resultados/<nombre>/`:

- `metricas_val.json`: F1 macro, F1 micro, F1 por película, mAP, acierto exacto, F1 de los
  cuatro géneros débiles y la línea base trivial.
- `por_genero_val.csv`: `genero, genero_es, peliculas, precision, recall, F1, umbral`.
- `registro_val.csv`: una fila por evaluación (fecha, nombre, pareja, split, F1 macro, F1 micro,
  mAP, F1 de los cuatro débiles, vectores, commit).

Luego vuelve a armar `resultados/registro.csv` juntando los `registro_*.csv` de todos los
experimentos, por fecha. Esa es la tabla "base contra cada intento" del reporte. No va en git
(se arma, no se edita): así los Pull Requests de cada pareja no chocan.
`python -m clasificador.evaluar --registro` la arma sin evaluar nada.

**Test no se usa para elegir.** `evaluar` se niega a medir test fuera de un release.

### Contrato 4 — Modelo combinado: ensamble y apilamiento (lo produce C; lo usan todos)

Junta las cabezas de varios experimentos **ya entrenados** (los *miembros*), promedia sus 17
probabilidades con pesos, opcionalmente las corrige con una etapa 2 (apilamiento) y aplica sus
propios umbrales. Se arma con `clasificador.combinado.armar(...)` y se guarda con
`combinado.guardar(...)` en `<P2M_SALIDAS>/modelos/<nombre>.pt`:

```python
{
  "tipo": "combinado",
  "generos": [...17...],
  "miembros": [                       # uno por experimento, sacado de su .pt (contrato 2)
    {"nombre": "base",     "vectores": ["clip_b32__centro"], "arquitectura": {...}, "capa": {...}},
    {"nombre": "A_l14",    "vectores": ["clip_l14__centro"], "arquitectura": {...}, "capa": {...}},
  ],
  "pesos": [0.5, 0.5],                # suman 1 (armar los normaliza)
  "apilamiento": None,                # o {"arquitectura": {...}, "capa": {...}, "residual": True}
  "umbrales": [...17...],             # calibrados en val sobre la salida FINAL
  "vectores": [...],                  # todos los archivos que usa (informativo)
  "experimento": {...}, "commit": "..."
}
```

- **Ensamble (7b):** varios miembros, `apilamiento: None`.
- **Apilamiento (6):** la etapa 2 recibe las 17 probabilidades promediadas. Con `residual: True`
  (recomendado) su salida se suma al logit del promedio: solo ajusta lo que ya había. Sobre un
  solo modelo = un miembro con peso 1.
- **Cómo combinar está definido en un solo lugar** (`combinado.probabilidades`): lo usan
  `evaluar` (con vectores) y `predictor` (con carteles), así dan lo mismo. Un `.pt` normal se
  predice como un combinado de un miembro.
- Un combinado se evalúa, compara y publica como cualquier experimento
  (`python -m clasificador.evaluar C_ensamble`).
- El predictor crea **un codificador por archivo de vectores**, compartido entre miembros, y un
  solo CLIP por modelo aunque lo usen varias variantes: revisen la memoria de la T4 (si no cabe,
  publiquen también una versión ligera).
