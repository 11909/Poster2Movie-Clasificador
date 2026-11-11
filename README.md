# Poster2Movie — Clasificador

Módulo 1 de [Poster2Movie](https://github.com/AndreaHortaHernandez/poster2movie): recibe el
**cartel** de una película y regresa sus **géneros** (17 posibles, multi-etiqueta).

Aquí el equipo del clasificador mejora el sistema base, con cada modelo publicado como
**release** (su `.pt`, sus métricas y la receta para reproducirlo). La meta es subir el F1 de
los géneros más débiles sin bajar el F1 macro:

| Género | F1 base (test) |
| --- | --- |
| Misterio | 0.27 |
| Historia | 0.33 |
| Fantasía | 0.36 |
| Crimen | 0.43 |
| **F1 macro (17 géneros)** | **0.537** |

## Estructura

```
poster2movie/            repo principal como submódulo (fijo en un commit): p2m.config, p2m.datos
clasificador/
├── codificadores.py     Pareja A · cartel -> vector (contrato 1)
├── cabezas.py           Pareja B · lineal, MLP, ...
├── perdidas.py          Pareja B · BCE, Focal Loss
├── zeroshot.py          Pareja B · similitudes con frases por género
├── apilamiento.py       Pareja C · etapa 2, predicciones fuera de pliegue
├── ensamble.py          Pareja C · promedio de probabilidades
├── entrenar.py          contrato 2
├── evaluar.py           contrato 3
├── comparar.py          modelo contra modelo, género por género
├── release.py           arma releases/<tag>/
├── predictor.py         Clasificador().predecir(), misma interfaz que el repo principal
├── datos.py, rutas.py   lectura de vectores y etiquetas, carpetas
experimentos/            un .yaml por experimento
resultados/              métricas de cada experimento + registro.csv (sí van en git)
releases/                una carpeta por release (sí van en git, .pt incluido)
notebooks/               Colab
tests/                   compatibilidad con el repo principal y flujo completo
```

## Empezar

Abre [`notebooks/01_reproducir_base.ipynb`](notebooks/01_reproducir_base.ipynb) en Colab. En
resumen:

```python
import os
os.environ["P2M_DATOS"] = "/content/drive/MyDrive/poster2movie_datos"          # dataset compartido, solo lectura
os.environ["P2M_SALIDAS"] = "/content/drive/MyDrive/poster2movie_clasificador" # carpeta del equipo
```

```bash
git clone --recurse-submodules https://github.com/11909/Poster2Movie-Clasificador.git
cd Poster2Movie-Clasificador
pip install -r requirements.txt
python -m clasificador.entrenar experimentos/base.yaml
python -m clasificador.evaluar base
```

Si clonaste sin `--recurse-submodules`, corre `git submodule update --init`.

## Cómo se hace un experimento

1. Rama nueva: `git checkout -b B-focal-loss`.
2. Copia `experimentos/base.yaml` a `experimentos/<pareja>_<idea>.yaml` (p. ej. `B_focal_g2.yaml`)
   y cambia lo que pruebas. El campo `nombre` debe ser igual al nombre del archivo.
3. `python -m clasificador.entrenar experimentos/B_focal_g2.yaml`
4. `python -m clasificador.evaluar B_focal_g2` (en validación).
5. `python -m clasificador.comparar base B_focal_g2`: los cuatro géneros débiles salen primero.
6. Sube el `.yaml`, el código y `resultados/` en un Pull Request. Lo revisa alguien de **otra**
   pareja. Aunque el experimento no mejore, se registra: explicar por qué no funcionó también cuenta.

---

## Contratos

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

Un tipo nuevo de cabeza o de pérdida se agrega en `cabezas.py` o `perdidas.py` y se documenta aquí.

### Contrato 3 — Métricas (lo usan todos)

`python -m clasificador.evaluar <nombre>` escribe en `resultados/<nombre>/`:

- `metricas_val.json`: F1 macro, F1 micro, F1 por película, mAP, acierto exacto, F1 de los
  cuatro géneros débiles y la línea base trivial.
- `por_genero_val.csv`: `genero, genero_es, peliculas, precision, recall, F1, umbral`.

Y agrega una fila a `resultados/registro.csv` (fecha, nombre, pareja, split, F1 macro, F1 micro,
mAP, F1 de los cuatro débiles, vectores, commit). Ese archivo es la tabla "base contra cada
intento" del reporte.

**Test no se usa para elegir.** `evaluar` se niega a medir test fuera de un release.

---

## Versiones y releases

`vMAYOR.MENOR.PARCHE`:

- **PARCHE** (`v0.2.1`): se arregla un error sin cambiar el modelo ni sus métricas.
- **MENOR** (`v0.3.0`): un modelo nuevo que mejora en validación.
- **MAYOR** (`v1.0.0`): la versión que se entrega al repo principal.

Se publica cuando el F1 macro en validación no baja y el promedio de los cuatro géneros débiles
sube:

```bash
python -m clasificador.release v0.2.0 B_focal_g2 --notas "Focal Loss gamma 2: +0.04 en Misterio"
git add releases/v0.2.0 resultados && git commit -m "Release v0.2.0"
git tag v0.2.0 && git push && git push --tags
```

`release.py` mide en val y test, copia el `.pt`, las métricas y la receta a `releases/v0.2.0/`, y
escribe `LEEME.md` con la comparación contra el release anterior. Al subir la etiqueta, GitHub
Actions crea el release con esos archivos. Un release no se reescribe: si algo salió mal, se
publica el siguiente número.

## Reglas

1. No se cambian el split ni los géneros: vienen del submódulo, y `tests/test_compatibilidad.py`
   falla si cambian. El submódulo solo se actualiza por acuerdo del equipo.
2. La interfaz `Clasificador().predecir(imagen)` no cambia: la usan la demo y los otros módulos.
3. Los modelos se eligen con validación. Test solo en releases.
4. Nada se sube directo a `main`: rama + Pull Request revisado por otra pareja. Las pruebas
   corren solas en cada PR.

## Entrega al repo principal

Al final se abre un PR al repo principal con el `predictor.py` final como `p2m/clasificador.py`
(o el paquete completo) y el `.pt` de `v1.0.0`. Los `.pt` de aquí ya tienen las claves que espera
el evaluador del repo principal.
