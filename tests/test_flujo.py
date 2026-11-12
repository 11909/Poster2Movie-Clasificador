"""El flujo completo con un dataset falso: entrenar -> evaluar -> comparar -> release.

No necesita GPU, ni Drive, ni descargar modelos.
"""

import json

import numpy as np
import pandas as pd
import pytest

import clasificador  # noqa: F401
from clasificador import rutas
from p2m import config

N = 600


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    datos_dir, salidas_dir = tmp_path / "datos", tmp_path / "salidas"
    datos_dir.mkdir()
    rng = np.random.default_rng(0)
    ids = np.arange(1, N + 1)
    x = rng.normal(size=(N, 32)).astype(np.float32)
    # géneros que dependen de los vectores, para que haya algo que aprender
    pesos = rng.normal(size=(32, len(config.GENEROS)))
    marcas = (x @ pesos) > 1.0
    marcas[np.arange(N), (x @ pesos).argmax(axis=1)] = True
    generos = [[g for g, m in zip(config.GENEROS, fila) if m] for fila in marcas]
    split = np.array(["train"] * (N * 8 // 10) + ["val"] * (N // 10) + ["test"] * (N // 10))
    pd.DataFrame({"id": ids, "generos": generos, "split": split}).to_parquet(datos_dir / "metadata.parquet")
    np.savez(datos_dir / "embeddings_clip_b32.npz", ids=ids, vectores=x)  # el alias de la base

    monkeypatch.setenv("P2M_DATOS", str(datos_dir))
    monkeypatch.setenv("P2M_SALIDAS", str(salidas_dir))
    monkeypatch.setattr(rutas, "RESULTADOS", tmp_path / "resultados")
    monkeypatch.setattr(rutas, "REGISTRO", tmp_path / "resultados" / "registro.csv")
    monkeypatch.setattr(rutas, "RELEASES", tmp_path / "releases")
    return tmp_path


def escribir_experimento(carpeta, nombre, cabeza):
    ruta = carpeta / f"{nombre}.yaml"
    ruta.write_text(
        f"nombre: {nombre}\npareja: base\nvectores: [clip_b32__centro]\ncabeza: {json.dumps(cabeza)}\n"
        "entrenamiento: {epocas: 30, paciencia: 5}\n"
    )
    return ruta


def test_flujo_completo(entorno):
    from clasificador.comparar import comparar
    from clasificador.entrenar import cargar_experimento, entrenar
    from clasificador.evaluar import evaluar
    from clasificador.release import preparar

    base = escribir_experimento(entorno, "base", {"tipo": "lineal"})
    mlp = escribir_experimento(entorno, "B_mlp", {"tipo": "mlp", "ocultas": [64]})
    for ruta in (base, mlp):
        modelo = entrenar(cargar_experimento(ruta))
        assert modelo.exists()
        assert (entorno / "salidas" / "predicciones" / f"{ruta.stem}.npz").exists()

    resultado = evaluar("base", "val")
    assert resultado["peliculas"] == N // 10
    assert resultado["F1_macro"] > resultado["trivial"]["F1_macro"]
    evaluar("B_mlp", "val")

    # cada experimento escribe solo en su carpeta; registro.csv junta las filas de todos
    for nombre in ("base", "B_mlp"):
        assert len(pd.read_csv(entorno / "resultados" / nombre / "registro_val.csv")) == 1
    registro = pd.read_csv(entorno / "resultados" / "registro.csv")
    assert sorted(registro["nombre"]) == ["B_mlp", "base"]
    evaluar("base", "val")  # volver a evaluar agrega otra fila, en su propio archivo
    assert len(pd.read_csv(entorno / "resultados" / "base" / "registro_val.csv")) == 2
    assert len(pd.read_csv(entorno / "resultados" / "registro.csv")) == 3

    with pytest.raises(SystemExit):
        evaluar("base", "test")  # test solo en releases

    tabla = comparar("base", "B_mlp")
    assert len(tabla) == len(config.GENEROS)
    assert set(tabla.index[:4]) == {"Mystery", "History", "Fantasy", "Crime"}  # los débiles primero

    preparar("v0.1.0", "base", "prueba")
    preparar("v0.2.0", "B_mlp", "prueba")
    leeme = (entorno / "releases" / "v0.2.0" / "LEEME.md").read_text()
    assert "Contra v0.1.0" in leeme
    assert (entorno / "releases" / "v0.2.0" / "B_mlp.pt").exists()
    with pytest.raises(SystemExit):
        preparar("v0.2.0", "B_mlp")  # un release no se reescribe


def test_experimento_mal_nombrado(entorno):
    from clasificador.entrenar import cargar_experimento

    ruta = entorno / "otro.yaml"
    ruta.write_text("nombre: base\npareja: A\nvectores: [clip_b32__centro]\n")
    with pytest.raises(ValueError):
        cargar_experimento(ruta)
