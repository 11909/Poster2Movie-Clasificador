# Poster2Movie — Clasificador

Módulo 1 de [Poster2Movie](https://github.com/AndreaHortaHernandez/poster2movie): recibe el
**cartel** de una película y regresa sus **géneros** (17 posibles, multi-etiqueta). Aquí el equipo
mejora el sistema base; cada modelo bueno se publica como **release** y el repo principal lo usa.

**Meta:** subir el F1 de los géneros débiles sin bajar el F1 macro de la base.

| Género | F1 base (test) | Se confunde con |
| --- | --- | --- |
| Misterio | 0.27 | Thriller |
| Historia | 0.33 | Drama, Guerra |
| Fantasía | 0.36 | Aventura, Familia |
| Crimen | 0.43 | Thriller |
| **F1 macro (17 géneros)** | **0.537** | trivial: 0.065 |

**Base:** CLIP ViT-B/32 congelado (cartel → vector de 512) + una capa lineal (→ 17 probabilidades)
+ un umbral por género calibrado en validación.

---

## Quién hace qué

Tres parejas que trabajan en paralelo. Cada una tiene su guía con tareas, archivos, comandos y
cuándo está "listo":

| Pareja | Qué hace | Ideas | Guía |
| --- | --- | --- | --- |
| **A** · Qué ve el modelo | Vectores nuevos del cartel | 1 multi-recorte · 2 OCR · 7a nuevos codificadores | [docs/PAREJA_A.md](docs/PAREJA_A.md) |
| **B** · Cómo aprende | Cabeza, pérdida y pistas de texto | 3 MLP · 4 Focal Loss · 5 zero-shot | [docs/PAREJA_B.md](docs/PAREJA_B.md) |
| **C** · Combinar y publicar | Ensamble, apilamiento, releases e integración | 6 apilamiento · 7b ensamble | [docs/PAREJA_C.md](docs/PAREJA_C.md) |

Cómo se conectan: **A** produce vectores → **B** y **C** los usan · **C** combina los modelos de
A y B y publica. Los formatos que los unen están en [docs/CONTRATOS.md](docs/CONTRATOS.md).
B y C empiezan desde el día uno con los vectores B/32 que ya existen.

---

## Primera vez

1. **Accesos** (pídelos a quien administra el repo y la carpeta):
   - Colaborador del repo `11909/Poster2Movie-Clasificador` en GitHub.
   - `poster2movie_datos` (dataset, **solo lectura**) y `poster2movie_clasificador` (carpeta del
     equipo, **con edición**) en Google Drive. Agrega las dos a *Mi unidad* como acceso directo
     (clic derecho → *Organizar → Agregar acceso directo*).
2. **Token de GitHub** para subir desde Colab: *Settings → Developer settings → Personal access
   tokens → Tokens (classic)*, permiso `repo`. Guárdalo; los notebooks te lo piden al subir.
   (Los *fine-grained* no sirven: el repo es de otra cuenta personal.)
3. **Una sola persona, una sola vez:** abre `notebooks/00_reproducir_base.ipynb` en Colab y córrelo.
   Reproduce la base y publica `v0.1.0`; en test debe dar F1 macro **0.537**.
4. **Cada quien:** abre el notebook de su idea en Colab
   (`https://colab.research.google.com/github/11909/Poster2Movie-Clasificador/blob/main/notebooks/<notebook>.ipynb`)
   y corre la celda de preparación: monta Drive, clona el repo, se cambia a la rama de la idea e instala.
5. Programa en `clasificador/*.py` (ver tu guía), corre el notebook y sube con su última celda.
   Abre el Pull Request a `main`; lo revisa alguien de **otra** pareja.

En tu computadora (opcional, para programar y correr las pruebas):

```bash
git clone --recurse-submodules https://github.com/11909/Poster2Movie-Clasificador.git
cd Poster2Movie-Clasificador
pip install -r requirements.txt pytest
pytest -q -rxX                     # sin GPU ni Drive: dataset falso
```

Para entrenar en local, define `P2M_DATOS` y `P2M_SALIDAS` con las carpetas de Google Drive para
escritorio.

---

## Flujo de un experimento

1. Rama de la idea (`B-focal`): la crea la celda de preparación del notebook.
2. Un `.yaml` en `experimentos/<pareja><idea>_<algo>.yaml` (hay de partida para cada idea). El campo
   `nombre` = nombre del archivo.
3. `entrenar` → `evaluar` (validación) → `comparar base <nombre>` (los 4 débiles salen primero).
4. Sube el `.yaml`, el código y `resultados/<nombre>/` y abre el Pull Request. **Aunque no mejore,
   se registra**: explicar por qué no funcionó también cuenta.
5. Si mejora, la Pareja C lo publica como release.

## Notebooks

Uno por idea: dos personas editando el mismo `.ipynb` dan conflictos casi imposibles de fusionar.
La lógica vive en `clasificador/*.py`; el notebook solo la corre.

| Notebook | Pareja · idea | Qué hace | Necesita | GPU |
| --- | --- | --- | --- | --- |
| `00_reproducir_base` | todos | Reproduce la base y publica `v0.1.0`. **Una vez, antes que los demás** | nada | no |
| `A1_multirecorte` | A · 1 | Vectores `relleno` y `3recortes` + experimentos | nada | sí |
| `A2_ocr` | A · 2 | OCR en 2 mil carteles + experimento; si ayuda, todo el dataset | nada | sí |
| `A7_codificadores` | A · 7a | Vectores CLIP L/14 y SigLIP + una cabeza por codificador | nada | sí |
| `B3_cabezas` | B · 3 | Rejilla de MLP | vectores B/32 (ya existen) | no |
| `B4_focal` | B · 4 | Focal Loss con γ = 1, 2, 3 | vectores B/32 | no |
| `B5_zeroshot` | B · 5 | Pistas zero-shot + experimento imagen + pistas | vectores B/32 | una vez |
| `C6_apilamiento` | C · 6 | Etapa 2 fuera de pliegue + mapa de calor 17 × 17 | cualquier experimento | no |
| `C7_ensamble` | C · 7b | Ensamble contra cada modelo solo + memoria en T4 | experimentos de A7 (o B/32 mientras) | sí |
| `C_release` | C | Comparar, publicar un release y probar un cartel | lo que se publique | no |

Todos empiezan con la misma celda de preparación y terminan con la misma celda para **subir a la
rama de la idea** (nunca a `main`). En Colab el repo se borra al cerrar: suban `resultados/` antes.

## Cómo saber qué falta

- `pytest -q -rxX`: cada idea tiene su prueba en `tests/test_pendientes.py`, marcada como pendiente
  (`x`). Cuando la idea funciona, pytest avisa (`XPASS`) y se quita la marca en ese mismo PR.
- Lo pendiente **avisa, nunca corre en silencio**: pedir `clip_l14`, `--variante ocr`,
  `activacion: gelu` o `perdida: focal` antes de implementarlos da un error que dice qué pareja lo
  hace y dónde.
- `resultados/registro.csv` (se arma con `python -m clasificador.evaluar --registro`): todos los
  experimentos con sus métricas en validación. Es la tabla "base contra cada intento" del reporte.

---

## Glosario de comandos

Todos se corren desde la raíz del repo (en Colab, con `!` al inicio).

**Preparar**

| Comando | Qué hace |
| --- | --- |
| `git clone --recurse-submodules https://github.com/11909/Poster2Movie-Clasificador.git` | Clona el repo con el repo principal como submódulo (`poster2movie/`) |
| `git submodule update --init` | Trae el submódulo si clonaste sin `--recurse-submodules` |
| `pip install -r requirements.txt` | Instala las dependencias |
| `pytest -q -rxX` | Corre las pruebas y lista las ideas pendientes |
| `pytest -q tests/test_pendientes.py -k idea4` | Solo la prueba de una idea |

**Vectores (contrato 1)**

| Comando | Qué hace |
| --- | --- |
| `python -m clasificador.codificadores clip_b32 --variante centro` | Vectores de todo el dataset → `<P2M_SALIDAS>/vectores/clip_b32__centro.npz` |
| `python -m clasificador.codificadores <codificador> --variante <variante>` | Igual con otro codificador (`clip_l14`, `siglip_base`…) o variante (`relleno`, `3recortes`, `ocr`) |
| `... --muestra 2000` | Solo 2 mil películas fijas (prueba chica) |
| `python -m clasificador.zeroshot clip_b32` | Pistas zero-shot → `vectores/clip_b32__zeroshot.npz` |

**Entrenar y medir (contratos 2 y 3)**

| Comando | Qué hace |
| --- | --- |
| `python -m clasificador.entrenar experimentos/<nombre>.yaml` | Entrena → `<P2M_SALIDAS>/modelos/<nombre>.pt` y `predicciones/<nombre>.npz` |
| `python -m clasificador.evaluar <nombre>` | Métricas en **validación** → `resultados/<nombre>/` y registro |
| `python -m clasificador.evaluar --registro` | Solo arma `resultados/registro.csv` con todos los experimentos |
| `python -m clasificador.comparar <a> <b>` | Dos experimentos o releases, género por género (débiles primero) |
| `python -m clasificador.comparar v0.1.0 v0.2.0 --split test` | Dos releases en test |

**Combinar (contrato 4)**

| Comando | Qué hace |
| --- | --- |
| `python -m clasificador.ensamble <nombre> <miembro1> <miembro2> ... [--pesos ...]` | Ensamble → `modelos/<nombre>.pt` |
| `python -m clasificador.apilamiento <nombre> <miembro> ... [--pliegues 5]` | Apilamiento → `modelos/<nombre>.pt` |

**Publicar**

| Comando | Qué hace |
| --- | --- |
| `git checkout -b release-v0.2.0` | Rama del release |
| `python -m clasificador.release v0.2.0 <nombre> --notas "..."` | Mide val y test y arma `releases/v0.2.0/` (`.pt`, métricas, receta, `LEEME.md`) |
| `git tag v0.2.0 && git push origin v0.2.0` | Después de fusionar el PR, en `main`: GitHub Actions publica el release |

**Usar el modelo (Python)**

```python
from clasificador.predictor import Clasificador
Clasificador().predecir(imagen_pil)            # el release más reciente de releases/
Clasificador("ruta/modelo.pt").predecir(img)   # un .pt normal o combinado
# -> {"generos": ["Horror", ...], "probabilidades": {"Action": 0.12, ...}}
```

## Glosario de términos

| Término | Qué es |
| --- | --- |
| **Codificador** | Modelo ya entrenado (CLIP, SigLIP) que convierte el cartel en un vector de números. No se entrena aquí. |
| **Variante** | Cómo se prepara el cartel antes del codificador: `centro`, `relleno`, `3recortes`, `ocr`, `zeroshot`. |
| **Vectores** | Archivo `<codificador>__<variante>.npz` con el vector de cada película (contrato 1). |
| **Cabeza** | La parte que entrenamos: recibe el vector y da 17 probabilidades. |
| **Umbral** | Por género, desde qué probabilidad se dice "sí". Se calibra en validación. |
| **Experimento** | Un `.yaml` (vectores + cabeza + pérdida) y su `.pt` entrenado (contrato 2). |
| **Combinado** | `.pt` que junta varias cabezas: ensamble y/o apilamiento (contrato 4). |
| **F1 macro** | Promedio del F1 de los 17 géneros sin pesar por frecuencia. La métrica principal. |
| **Validación / test** | Validación: con la que se elige. Test: solo se mide al publicar un release. |
| **Release** | Versión publicada (`vX.Y.Z`) con su `.pt`, métricas y receta. La usa el repo principal. |
| **`P2M_DATOS` / `P2M_SALIDAS`** | Carpeta del dataset (solo lectura) / carpeta compartida del equipo en Drive. |

---

## Estructura

```
poster2movie/            repo principal como submódulo (fijo en un commit): p2m.config, p2m.datos
clasificador/
├── codificadores.py     A · cartel -> vector (contrato 1); B agrega la variante zeroshot
├── cabezas.py           B · lineal, MLP...
├── perdidas.py          B · BCE, Focal Loss
├── zeroshot.py          B · similitudes con frases por género
├── apilamiento.py       C · etapa 2, predicciones fuera de pliegue
├── ensamble.py          C · promedio de probabilidades
├── combinado.py         contrato 4: cómo se combinan (lo usan evaluar y predictor)
├── entrenar.py          contrato 2
├── evaluar.py           contrato 3
├── comparar.py          modelo contra modelo, género por género
├── release.py           arma releases/<tag>/
├── predictor.py         Clasificador().predecir(), misma interfaz que el repo principal
├── datos.py, rutas.py   lectura de vectores y etiquetas, carpetas
docs/                    guía de cada pareja y contratos
experimentos/            un .yaml por experimento
resultados/              una carpeta por experimento (sí va en git); registro.csv se arma (no va)
releases/                una carpeta por release (sí va en git, .pt incluido)
notebooks/               Colab: uno por idea
tests/                   compatibilidad, flujo completo, combinado, experimentos e ideas pendientes
```

En Drive, `poster2movie_clasificador/` (`P2M_SALIDAS`) guarda lo pesado, que no va en git:
`vectores/`, `modelos/` y `predicciones/`.

## Reglas

1. No se cambian el split ni los géneros: vienen del submódulo, y `tests/test_compatibilidad.py`
   falla si cambian. El submódulo solo se actualiza por acuerdo del equipo.
2. La interfaz `Clasificador().predecir(imagen)` no cambia: la usan la demo y los otros módulos.
3. Los modelos se eligen con **validación**. Test solo en releases.
4. Nada se sube directo a `main`: rama + Pull Request revisado por otra pareja. Las pruebas
   corren solas en cada PR.
5. Los contratos ([docs/CONTRATOS.md](docs/CONTRATOS.md)) solo cambian por acuerdo del equipo y se
   anotan en `CHANGELOG.md`.
6. Se publica un release cuando el F1 macro en validación no baja y el promedio de los 4 débiles sube.
