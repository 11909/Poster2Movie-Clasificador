"""Modelos combinados: ensamble (idea 7b) y apilamiento (idea 6) — contrato 4.

Un `.pt` combinado junta las cabezas de varios experimentos ya entrenados (los "miembros",
cada uno con sus `vectores`), promedia sus probabilidades con pesos y, si tiene apilamiento,
las corrige con una etapa 2. Al final, sus propios umbrales calibrados en val.

    cartel ─┬─ codificadores de A ─ cabeza A ─ 17 prob ─┐
            ├─ codificadores de B ─ cabeza B ─ 17 prob ─┼─ promedio con pesos ─ [etapa 2] ─ umbrales ─ géneros
            └─ ...                                       ┘

Este archivo es la ÚNICA definición de cómo se combinan: lo usan `evaluar` (con vectores
precalculados) y `predictor` (con carteles), así los dos dan lo mismo. La Pareja C decide
QUÉ miembros, pesos y etapa 2 (ensamble.py, apilamiento.py) y lo guarda con `guardar`.

Apilamiento sobre UN solo modelo = un combinado de un miembro con peso 1.
"""

from pathlib import Path

import numpy as np
import torch

from clasificador.cabezas import construir_cabeza
from clasificador.datos import GENEROS

TIPO = "combinado"
CLAVES_MIEMBRO = ("nombre", "vectores", "arquitectura", "capa")


def es_combinado(guardado: dict) -> bool:
    return guardado.get("tipo") == TIPO


def miembro(guardado: dict) -> dict:
    """Lo que se guarda de un .pt normal (contrato 2) para usarlo dentro de un combinado."""
    if es_combinado(guardado):
        raise ValueError("Un combinado no puede ser miembro de otro: usa sus miembros directamente.")
    return {
        "nombre": (guardado.get("experimento") or {}).get("nombre", ""),
        "vectores": list(guardado.get("vectores") or ["clip_b32__centro"]),
        "arquitectura": guardado.get("arquitectura") or {"tipo": "lineal"},
        "capa": guardado["capa"],
    }


def armar(miembros: list[dict], umbrales, pesos=None, apilamiento: dict | None = None,
          experimento: dict | None = None, commit: str = "") -> dict:
    """El diccionario del contrato 4.

    miembros     .pt normales (los `torch.load` de cada experimento) o ya pasados por `miembro`
    umbrales     17 umbrales calibrados en val sobre la salida FINAL (después de la etapa 2)
    pesos        uno por miembro; se normalizan para sumar 1. None = promedio simple
    apilamiento  None o {"arquitectura": {...}, "capa": state_dict, "residual": True}:
                 etapa 2 que recibe las 17 probabilidades promediadas. Con residual=True
                 (recomendado) su salida se SUMA al logit del promedio: solo ajusta.
    """
    # un .pt completo trae umbrales (los suyos, que aquí no sirven); un miembro ya armado no
    miembros = [miembro(m) if "umbrales" in m else m for m in miembros]
    for m in miembros:
        if not set(CLAVES_MIEMBRO) <= m.keys():
            raise ValueError(f"A un miembro le faltan claves: {set(CLAVES_MIEMBRO) - m.keys()}")
    if not miembros:
        raise ValueError("Un combinado necesita al menos un miembro.")
    pesos = np.ones(len(miembros)) if pesos is None else np.asarray(pesos, dtype=float)
    if len(pesos) != len(miembros) or (pesos < 0).any() or pesos.sum() <= 0:
        raise ValueError("Debe haber un peso >= 0 por miembro, y no todos en cero.")
    umbrales = np.asarray(umbrales, dtype=float)
    if umbrales.shape != (len(GENEROS),):
        raise ValueError(f"Se esperan {len(GENEROS)} umbrales.")
    if apilamiento is not None:
        apilamiento = {"residual": True, **apilamiento}
    vectores = list(dict.fromkeys(v for m in miembros for v in m["vectores"]))
    return {
        "tipo": TIPO, "generos": GENEROS, "miembros": miembros,
        "pesos": (pesos / pesos.sum()).tolist(), "apilamiento": apilamiento, "umbrales": umbrales,
        "vectores": vectores,                 # todos los que usa (informativo)
        "modelo_clip": None, "embeddings": None,  # el repo principal los evalúa codificando carteles
        "experimento": experimento or {}, "commit": commit,
    }


def guardar(combinado: dict, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combinado, ruta)
    return ruta


def cargar_cabezas(guardado: dict, dims: dict[str, int]) -> list[torch.nn.Module]:
    """Una cabeza por miembro. `dims`: dimensión de cada archivo de vectores."""
    cabezas = []
    for m in guardado["miembros"]:
        cabeza = construir_cabeza(m["arquitectura"], sum(dims[v] for v in m["vectores"]), len(GENEROS))
        cabeza.load_state_dict(m["capa"])
        cabezas.append(cabeza.eval())
    return cabezas


def cargar_etapa2(guardado: dict) -> torch.nn.Module | None:
    if not guardado.get("apilamiento"):
        return None
    etapa2 = construir_cabeza(guardado["apilamiento"]["arquitectura"], len(GENEROS), len(GENEROS))
    etapa2.load_state_dict(guardado["apilamiento"]["capa"])
    return etapa2.eval()


@torch.no_grad()
def probabilidades(guardado: dict, cabezas: list, etapa2, vectores: dict[str, np.ndarray]) -> np.ndarray:
    """Probabilidades finales (n × 17) a partir de los vectores de cada archivo (n × D)."""
    promedio = 0.0
    for m, cabeza, peso in zip(guardado["miembros"], cabezas, guardado["pesos"]):
        x = torch.tensor(np.concatenate([vectores[v] for v in m["vectores"]], axis=1), dtype=torch.float32)
        promedio = promedio + peso * torch.sigmoid(cabeza(x))
    if etapa2 is None:
        return promedio.numpy()
    ajuste = etapa2(promedio)
    if guardado["apilamiento"].get("residual", True):
        ajuste = ajuste + torch.logit(promedio.clamp(1e-6, 1 - 1e-6))
    return torch.sigmoid(ajuste).numpy()
