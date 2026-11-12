"""Vectores + etiquetas por split, y los umbrales por género.

Port directo de `p2m/clasificador.py` (cargar_split, calibrar_umbrales,
aplicar_umbrales), con un cambio: acepta VARIOS archivos de vectores y los
concatena por id (por ejemplo imagen + texto OCR + similitudes zero-shot).
"""

import numpy as np
from sklearn.metrics import f1_score

from clasificador import rutas
from p2m import config, datos


def cargar_vectores(nombre: str) -> tuple[np.ndarray, np.ndarray]:
    """(ids, vectores) de un archivo del contrato 1."""
    guardado = np.load(rutas.ruta_vectores(nombre))
    return guardado["ids"].astype(np.int64), guardado["vectores"].astype(np.float32)


def cargar_split_por_archivo(split: str, vectores: list[str] | str) -> tuple[dict[str, np.ndarray], np.ndarray, list[int]]:
    """({archivo: x}, y, ids) de un split, con las filas alineadas entre archivos.
    Solo entran las películas que están en TODOS los archivos."""
    if isinstance(vectores, str):
        vectores = [vectores]
    tablas = {}  # archivo -> (fila de cada id, matriz de vectores)
    for nombre in dict.fromkeys(vectores):
        ids, x = cargar_vectores(nombre)
        tablas[nombre] = ({int(i): n for n, i in enumerate(ids)}, x)

    meta = datos.metadata(split)
    comunes = set(meta["id"])
    for fila_de, _ in tablas.values():
        comunes &= fila_de.keys()
    meta = meta[meta["id"].isin(comunes)]

    xs = {nombre: x[[fila_de[i] for i in meta["id"]]] for nombre, (fila_de, x) in tablas.items()}
    y = np.array([datos.vector_generos(list(g)) for g in meta["generos"]], dtype=np.int64)
    return xs, y, meta["id"].tolist()


def cargar_split(split: str, vectores: list[str] | str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(x, y, ids) de un split. `x` concatena los vectores de cada archivo, en el orden dado.
    Solo entran las películas que están en TODOS los archivos."""
    if isinstance(vectores, str):
        vectores = [vectores]
    xs, y, ids = cargar_split_por_archivo(split, vectores)
    return np.concatenate([xs[v] for v in vectores], axis=1), y, ids


def calibrar_umbrales(y: np.ndarray, probabilidades: np.ndarray) -> np.ndarray:
    """Para cada género, el umbral (de 0.05 a 0.95) que maximiza su F1."""
    candidatos = np.arange(0.05, 0.96, 0.01)
    umbrales = []
    for g in range(y.shape[1]):
        f1s = [f1_score(y[:, g], probabilidades[:, g] >= u, zero_division=0) for u in candidatos]
        umbrales.append(candidatos[int(np.argmax(f1s))])
    return np.array(umbrales)


def aplicar_umbrales(probabilidades: np.ndarray, umbrales: np.ndarray) -> np.ndarray:
    """Género presente si supera su umbral; si ninguno lo supera, el más probable."""
    prediccion = probabilidades >= umbrales
    vacias = ~prediccion.any(axis=1)
    prediccion[vacias, probabilidades[vacias].argmax(axis=1)] = True
    return prediccion.astype(int)


GENEROS = config.GENEROS
GENEROS_DEBILES = ["Mystery", "History", "Fantasy", "Crime"]
