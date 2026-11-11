"""Que el repo siga siendo comparable con el repo principal.

Si una de estas falla después de actualizar el submódulo, el repo principal cambió
algo que rompe la comparación con la base: avisen a todo el equipo antes de seguir.
"""

import numpy as np
import torch

import clasificador  # noqa: F401  (agrega el submódulo al path)
from clasificador.cabezas import cargar_cabeza, construir_cabeza
from clasificador.datos import aplicar_umbrales
from p2m import config

GENEROS = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War", "Western",
]


def test_generos_sin_cambios():
    assert config.GENEROS == GENEROS


def test_split_sin_cambios():
    esperados = {
        "pelicula_550": "train", "pelicula_603": "train", "coleccion_10": "train",
        "pelicula_13": "val", "pelicula_38": "val", "pelicula_49": "val",
        "pelicula_11": "test", "pelicula_17": "test", "pelicula_20": "test",
    }
    assert {g: config.split_de(g) for g in esperados} == esperados
    assert (config.PORCENTAJE_VAL, config.PORCENTAJE_TEST) == (10, 10)


def test_cabezas_y_carga():
    for arquitectura in ({"tipo": "lineal"}, {"tipo": "mlp", "ocultas": [64, 32], "dropout": 0.1}):
        cabeza = construir_cabeza(arquitectura, 512, 17)
        assert cabeza(torch.zeros(4, 512)).shape == (4, 17)
        guardado = {"arquitectura": arquitectura, "generos": GENEROS, "capa": cabeza.state_dict()}
        assert cargar_cabeza(guardado, 512)(torch.zeros(1, 512)).shape == (1, 17)
    # un .pt viejo sin arquitectura es lineal
    lineal = construir_cabeza(None, 512, 17)
    assert isinstance(lineal, torch.nn.Linear)


def test_toda_pelicula_tiene_un_genero():
    prob = np.array([[0.1, 0.2, 0.05], [0.9, 0.1, 0.8]])
    prediccion = aplicar_umbrales(prob, np.array([0.5, 0.5, 0.5]))
    assert prediccion.tolist() == [[0, 1, 0], [1, 0, 1]]
