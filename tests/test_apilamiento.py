"""Apilamiento (idea 6) con el dataset falso: pliegues por franquicia -> fuera de pliegue ->
etapa 2 -> combinado que evalúan `evaluar` y el predictor.

No necesita GPU, ni Drive, ni descargar modelos.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from clasificador import rutas
from p2m import config


def con_colecciones(entorno):
    """Agrega la columna `coleccion` a la metadata falsa: sagas de 1 a 6 películas."""
    ruta = entorno / "datos" / "metadata.parquet"
    meta = pd.read_parquet(ruta)
    rng = np.random.default_rng(1)
    meta["coleccion"] = [str(100 + i % 40) if rng.random() < 0.4 else "" for i in range(len(meta))]
    meta.to_parquet(ruta)
    return meta


def test_pliegues_balanceados_y_deterministas():
    from clasificador.apilamiento import pliegues

    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"id": range(1000),
                         "coleccion": [f"{rng.integers(50)}" if rng.random() < 0.3 else None for _ in range(1000)]})
    p = pliegues(meta, k=5)
    assert list(p.index) == list(meta.index)
    tamanos = p.value_counts()
    assert set(tamanos.index) == set(range(5)) and tamanos.max() - tamanos.min() <= tamanos.max() * 0.1
    sagas = meta.assign(p=p)[meta["coleccion"].notna()].groupby("coleccion")["p"].nunique()
    assert (sagas == 1).all()
    # el pliegue de cada película no depende del orden de las filas
    revuelta = meta.sample(frac=1, random_state=3)
    assert pliegues(revuelta, k=5).sort_index().equals(p)
    # sin columna de colección: cada película sola
    assert set(pliegues(meta[["id"]], k=5)) == set(range(5))
    with pytest.raises(ValueError):
        pliegues(meta.head(3), k=5)


def test_etapa2_sin_evidencia_no_corrige():
    """Si la etapa 1 ya está bien calibrada y los géneros son independientes, la etapa 2 casi no mueve nada."""
    from clasificador.apilamiento import entrenar_etapa2

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=(2000, 17))
    y = (rng.random((2000, 17)) < p).astype(int)
    etapa2 = entrenar_etapa2(p[:1500], y[:1500], p[1500:], y[1500:])
    assert etapa2.weight.abs().max() < 0.5


def test_etapa2_aprende_coocurrencia():
    """Misterio (columna) empuja a Crimen (fila) cuando en los datos siempre van juntos."""
    from clasificador.apilamiento import entrenar_etapa2
    from clasificador.datos import GENEROS

    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random((n, 17)) < 0.2).astype(int)
    m, c = GENEROS.index("Mystery"), GENEROS.index("Crime")
    y[:, c] = y[:, m]                                # Crimen <=> Misterio
    p = np.clip(0.2 + 0.1 * rng.normal(size=(n, 17)), 0.02, 0.98)
    p[:, m] = np.where(y[:, m] == 1, 0.8, 0.1)        # Misterio lo ve bien; Crimen no
    etapa2 = entrenar_etapa2(p[:2400], y[:2400], p[2400:], y[2400:])
    assert etapa2.weight[c, m] > 1.0


def test_flujo_apilamiento(entorno, escribir_experimento):
    from clasificador import predictor
    from clasificador.apilamiento import apilar
    from clasificador.datos import cargar_split
    from clasificador.entrenar import cargar_experimento, entrenar
    from clasificador.evaluar import evaluar, probabilidades_split

    meta = con_colecciones(entorno)
    for nombre, vectores in [("base", ["clip_b32__centro"]), ("otro", ["otro__centro"])]:
        entrenar(cargar_experimento(escribir_experimento(entorno, nombre, {"tipo": "lineal"}, vectores)))

    ruta = apilar("C6_apilamiento", ["base", "otro"], k=4)
    guardado = torch.load(ruta, weights_only=False)
    assert guardado["tipo"] == "combinado" and guardado["pesos"] == [0.5, 0.5]
    assert guardado["apilamiento"]["residual"] is True
    assert guardado["apilamiento"]["arquitectura"] == {"tipo": "lineal"}
    assert guardado["experimento"]["pareja"] == "C" and guardado["experimento"]["pliegues"] == 4

    # fuera de pliegue: una fila por película de train, cada saga en un solo pliegue
    oof = np.load(entorno / "salidas" / "predicciones" / "C6_apilamiento_oof.npz")
    _, _, ids_train = cargar_split("train", ["clip_b32__centro"])
    assert sorted(oof["ids_train"]) == sorted(ids_train) and set(oof["pliegue"]) == set(range(4))
    sagas = (pd.DataFrame({"id": oof["ids_train"], "p": oof["pliegue"]})
             .merge(meta, on="id").query("coleccion != ''").groupby("coleccion")["p"].nunique())
    assert (sagas == 1).all()
    # y no son las del modelo que vio todo train
    x, _, _ = cargar_split("train", ["clip_b32__centro"])
    base = torch.load(rutas.ruta_modelo("base"), weights_only=False)
    cabeza = torch.nn.Linear(x.shape[1], 17)
    cabeza.load_state_dict(base["capa"])
    with torch.no_grad():
        completo = torch.sigmoid(cabeza(torch.tensor(x))).numpy()
    orden = np.argsort(oof["ids_train"])
    assert not np.allclose(oof["prob_train__base"][orden], completo[np.argsort(ids_train)], atol=1e-4)

    # la matriz 17 × 17 para el reporte
    pesos = pd.read_csv(entorno / "resultados" / "C6_apilamiento" / "pesos_etapa2.csv", index_col=0)
    assert pesos.shape == (17, 17) and list(pesos.index) == config.GENEROS
    assert np.allclose(pesos.to_numpy(), guardado["apilamiento"]["capa"]["weight"].numpy(), atol=1e-4)

    # se evalúa como cualquier experimento, y evaluar y el predictor dan lo mismo
    resultado = evaluar("C6_apilamiento", "val")
    assert resultado["F1_macro"] > resultado["trivial"]["F1_macro"] and resultado["pareja"] == "C"
    prob, _ = probabilidades_split(guardado, "val")
    _, _, ids_val = cargar_split("val", ["clip_b32__centro"])

    class Falso:
        def __init__(self, nombre):
            from clasificador.datos import cargar_vectores
            i, v = cargar_vectores(nombre)
            self.fila, self.x, self.dim = {int(a): n for n, a in enumerate(i)}, v, v.shape[1]

        def codificar(self, ids):
            return self.x[[self.fila[i] for i in ids]]

    original = predictor.crear_codificador
    predictor.crear_codificador = Falso
    try:
        assert np.allclose(predictor.Clasificador(ruta).probabilidades(ids_val), prob, atol=1e-5)
    finally:
        predictor.crear_codificador = original


def test_mapa_calor(entorno):
    pytest.importorskip("matplotlib")
    from clasificador.apilamiento import guardar_pesos, mapa_calor

    guardar_pesos("C6_prueba", np.eye(17))
    assert mapa_calor("C6_prueba").exists()


def test_apilar_errores(entorno, escribir_experimento):
    from clasificador.apilamiento import apilar
    from clasificador.ensamble import ensamblar
    from clasificador.entrenar import cargar_experimento, entrenar

    entrenar(cargar_experimento(escribir_experimento(entorno, "base", {"tipo": "lineal"})))
    with pytest.raises(ValueError):
        apilar("base", ["base"])                          # no pisar a un miembro
    with pytest.raises(FileNotFoundError):
        apilar("C6_falta", ["no_existe"])
    with pytest.raises(ValueError):
        apilar("C6_pesos", ["base"], pesos=[1, 2])
    entrenar(cargar_experimento(escribir_experimento(entorno, "otro", {"tipo": "lineal"}, ["otro__centro"])))
    ensamblar("C7_ensamble", ["base", "otro"])
    with pytest.raises(ValueError):
        apilar("C6_doble", ["C7_ensamble"])                # un combinado no es miembro
