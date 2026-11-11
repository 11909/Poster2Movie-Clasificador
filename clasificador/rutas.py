"""Dónde se leen y se guardan los archivos del clasificador.

- Lectura: primero la carpeta del equipo (P2M_SALIDAS), luego el dataset compartido (P2M_DATOS).
- Escritura: siempre la carpeta del equipo. El dataset compartido es solo de lectura.
- Los resultados (métricas pequeñas) van dentro del repo, en `resultados/`, para versionarlos.
"""

import os
from pathlib import Path

from clasificador import RAIZ
from p2m import config

# Vectores del sistema base que ya existen en el dataset compartido, con su nombre del contrato 1.
ALIAS_VECTORES = {"clip_b32__centro": "embeddings_clip_b32.npz"}

RESULTADOS = RAIZ / "resultados"
REGISTRO = RESULTADOS / "registro.csv"
RELEASES = RAIZ / "releases"


def salidas() -> Path:
    """Carpeta del equipo en Drive. Se exige explícita para no escribir en el dataset compartido."""
    ruta = os.environ.get("P2M_SALIDAS")
    if not ruta:
        raise RuntimeError(
            "Define P2M_SALIDAS con la carpeta del equipo en Drive, por ejemplo "
            "os.environ['P2M_SALIDAS'] = '/content/drive/MyDrive/poster2movie_clasificador'."
        )
    return Path(ruta)


def carpeta_vectores() -> Path:
    return salidas() / "vectores"


def carpeta_modelos() -> Path:
    return salidas() / "modelos"


def carpeta_predicciones() -> Path:
    return salidas() / "predicciones"


def ruta_vectores(nombre: str) -> Path:
    """`clip_b32__centro` -> el .npz donde esté: carpeta del equipo, dataset compartido o alias de la base."""
    candidatos = []
    if os.environ.get("P2M_SALIDAS"):
        candidatos.append(carpeta_vectores() / f"{nombre}.npz")
    candidatos.append(config.carpeta_datos() / "vectores" / f"{nombre}.npz")
    if nombre in ALIAS_VECTORES:
        candidatos.append(config.carpeta_datos() / ALIAS_VECTORES[nombre])
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    raise FileNotFoundError(f"No encuentro los vectores '{nombre}'. Busqué en: " + ", ".join(map(str, candidatos)))


def ruta_modelo(nombre_o_ruta: str) -> Path:
    """Un nombre de experimento (`base`) o una ruta a un .pt."""
    ruta = Path(nombre_o_ruta)
    if ruta.suffix == ".pt":
        return ruta
    return carpeta_modelos() / f"{nombre_o_ruta}.pt"
