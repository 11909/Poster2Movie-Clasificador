"""El clasificador que usan la demo y los demás módulos. Misma interfaz que el repo principal:

    clasificador = Clasificador()                 # el release más reciente de releases/
    clasificador = Clasificador("ruta/modelo.pt") # o un .pt en particular
    clasificador.predecir(imagen_pil)  ->  {"generos": ["Horror", ...], "probabilidades": {...}}

Acepta los dos formatos de .pt:
- normal (contrato 2): una cabeza sobre uno o varios archivos de vectores concatenados;
- combinado (contrato 4): ensamble y/o apilamiento de varias cabezas (`combinado.py`).

Un .pt normal se trata como un combinado de un solo miembro con peso 1: así hay UN solo
camino de predicción. Cada codificador se crea una vez aunque lo usen varios miembros
(memoria de la T4 en la demo).
"""

from pathlib import Path

import numpy as np
import torch

from clasificador import combinado, rutas
from clasificador.codificadores import crear_codificador
from clasificador.datos import aplicar_umbrales


def ultimo_release() -> Path:
    """El .pt del release con la versión más alta en releases/."""
    def version(carpeta: Path) -> tuple[int, ...]:
        try:
            return tuple(int(n) for n in carpeta.name.lstrip("v").split("."))
        except ValueError:
            return (-1,)

    carpetas = sorted((c for c in rutas.RELEASES.glob("v*") if c.is_dir() and list(c.glob("*.pt"))), key=version)
    if not carpetas:
        raise FileNotFoundError("No hay releases en releases/. Pasa la ruta a un .pt.")
    return next(carpetas[-1].glob("*.pt"))


def como_combinado(guardado: dict) -> dict:
    """Un .pt normal como combinado de un miembro (mismas probabilidades, mismos umbrales)."""
    if combinado.es_combinado(guardado):
        return guardado
    return combinado.armar([guardado], guardado["umbrales"], experimento=guardado.get("experimento"))


class Clasificador:
    def __init__(self, ruta=None):
        guardado = torch.load(ruta or ultimo_release(), map_location="cpu", weights_only=False)
        self.guardado = como_combinado(guardado)
        self.generos = self.guardado["generos"]
        self.umbrales = np.asarray(self.guardado["umbrales"])
        # un codificador por archivo de vectores, compartido entre miembros
        self.codificadores = {v: crear_codificador(v) for v in self.guardado["vectores"]}
        self.cabezas = combinado.cargar_cabezas(self.guardado, {v: c.dim for v, c in self.codificadores.items()})
        self.etapa2 = combinado.cargar_etapa2(self.guardado)

    def probabilidades(self, imagenes) -> np.ndarray:
        vectores = {v: c.codificar(imagenes) for v, c in self.codificadores.items()}
        return combinado.probabilidades(self.guardado, self.cabezas, self.etapa2, vectores)

    def predecir(self, imagen) -> dict:
        prob = self.probabilidades([imagen])[0]
        presentes = aplicar_umbrales(prob[None], self.umbrales)[0]
        return {
            "generos": [g for g, p in zip(self.generos, presentes) if p],
            "probabilidades": {g: float(p) for g, p in zip(self.generos, prob)},
        }
