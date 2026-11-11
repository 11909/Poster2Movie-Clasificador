"""El clasificador que usan la demo y los demás módulos. Misma interfaz que el repo principal:

    clasificador = Clasificador()                 # el release más reciente de releases/
    clasificador = Clasificador("ruta/modelo.pt") # o un .pt en particular
    clasificador.predecir(imagen_pil)  ->  {"generos": ["Horror", ...], "probabilidades": {...}}

Hoy soporta modelos con UN archivo de vectores. Modelos que combinan varios
codificadores (ensamble, apilamiento) los agrega la Pareja C aquí, sin cambiar
la interfaz.
"""

from pathlib import Path

import numpy as np
import torch

from clasificador import rutas
from clasificador.cabezas import cargar_cabeza
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


class Clasificador:
    def __init__(self, ruta=None):
        guardado = torch.load(ruta or ultimo_release(), map_location="cpu", weights_only=False)
        vectores = guardado.get("vectores") or ["clip_b32__centro"]
        if len(vectores) != 1:
            raise NotImplementedError("Modelos con varios codificadores: pendiente de la Pareja C.")
        self.generos = guardado["generos"]
        self.umbrales = np.asarray(guardado["umbrales"])
        self.codificador = crear_codificador(vectores[0])
        self.capa = cargar_cabeza(guardado, self.codificador.dim)

    def probabilidades(self, imagenes) -> np.ndarray:
        vectores = torch.tensor(self.codificador.codificar(imagenes))
        with torch.no_grad():
            return torch.sigmoid(self.capa(vectores)).numpy()

    def predecir(self, imagen) -> dict:
        prob = self.probabilidades([imagen])[0]
        presentes = aplicar_umbrales(prob[None], self.umbrales)[0]
        return {
            "generos": [g for g, p in zip(self.generos, presentes) if p],
            "probabilidades": {g: float(p) for g, p in zip(self.generos, prob)},
        }
