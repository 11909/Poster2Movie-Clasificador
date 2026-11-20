"""Ensamble (idea 7b) con el dataset falso: entrenar dos miembros -> ensamblar -> evaluar.

No necesita GPU, ni Drive, ni descargar modelos.
"""

import numpy as np
import pytest
import torch


def test_alinear_ids_distintos():
    from clasificador.ensamble import alinear

    p = np.arange(4 * 17, dtype=float).reshape(4, 17)
    # "b" no tiene la película 4 y trae las filas en otro orden
    ids, (pa, pb) = alinear({"a": (np.array([1, 2, 3, 4]), p), "b": (np.array([3, 1, 2]), p[[2, 0, 1]])})
    assert list(ids) == [1, 2, 3]
    assert np.array_equal(pa, p[:3]) and np.array_equal(pb, p[:3])
    with pytest.raises(ValueError):
        alinear({"a": (np.array([1, 1]), p[:2])})            # ids repetidos
    with pytest.raises(ValueError):
        alinear({"a": (np.array([1]), p[:1]), "b": (np.array([2]), p[:1])})  # nada en común


def test_flujo_ensamble(entorno, escribir_experimento):
    from clasificador.ensamble import ensamblar
    from clasificador.entrenar import cargar_experimento, entrenar
    from clasificador.evaluar import evaluar

    for nombre, vectores in [("base", ["clip_b32__centro"]), ("otro", ["otro__centro"])]:
        entrenar(cargar_experimento(escribir_experimento(entorno, nombre, {"tipo": "lineal"}, vectores)))

    ruta = ensamblar("C7_ensamble", ["base", "otro"])
    guardado = torch.load(ruta, weights_only=False)
    assert guardado["tipo"] == "combinado" and guardado["apilamiento"] is None
    assert [m["nombre"] for m in guardado["miembros"]] == ["base", "otro"]
    assert guardado["pesos"] == [0.5, 0.5]                    # promedio simple
    assert len(guardado["umbrales"]) == 17

    resultado = evaluar("C7_ensamble", "val")
    assert resultado["F1_macro"] > resultado["trivial"]["F1_macro"]

    with pytest.raises(ValueError):
        ensamblar("base", ["base", "otro"])                   # no pisar a un miembro
    with pytest.raises(ValueError):
        ensamblar("C7_doble", ["C7_ensamble", "base"])        # un combinado no es miembro
    with pytest.raises(FileNotFoundError):
        ensamblar("C7_falta", ["base", "no_existe"])
