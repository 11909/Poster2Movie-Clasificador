"""Dataset falso para las pruebas: no necesita GPU, ni Drive, ni descargar modelos."""

import json

import numpy as np
import pandas as pd
import pytest

import clasificador  # noqa: F401  (agrega el submódulo al path)
from clasificador import rutas
from p2m import config

N = 600


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """600 películas con vectores `clip_b32__centro` (32 números) y géneros que dependen de ellos,
    para que haya algo que aprender. Más un segundo archivo `otro__centro` (16 números)."""
    datos_dir, salidas_dir = tmp_path / "datos", tmp_path / "salidas"
    datos_dir.mkdir()
    rng = np.random.default_rng(0)
    ids = np.arange(1, N + 1)
    x = rng.normal(size=(N, 32)).astype(np.float32)
    pesos = rng.normal(size=(32, len(config.GENEROS)))
    marcas = (x @ pesos) > 1.0
    marcas[np.arange(N), (x @ pesos).argmax(axis=1)] = True
    generos = [[g for g, m in zip(config.GENEROS, fila) if m] for fila in marcas]
    split = np.array(["train"] * (N * 8 // 10) + ["val"] * (N // 10) + ["test"] * (N // 10))
    pd.DataFrame({"id": ids, "generos": generos, "split": split}).to_parquet(datos_dir / "metadata.parquet")
    np.savez(datos_dir / "embeddings_clip_b32.npz", ids=ids, vectores=x)  # el alias de la base
    (salidas_dir / "vectores").mkdir(parents=True)
    otro = (x[:, :16] + rng.normal(scale=0.5, size=(N, 16))).astype(np.float32)
    np.savez(salidas_dir / "vectores" / "otro__centro.npz", ids=ids[::-1], vectores=otro[::-1])  # otro orden

    monkeypatch.setenv("P2M_DATOS", str(datos_dir))
    monkeypatch.setenv("P2M_SALIDAS", str(salidas_dir))
    monkeypatch.setattr(rutas, "RESULTADOS", tmp_path / "resultados")
    monkeypatch.setattr(rutas, "REGISTRO", tmp_path / "resultados" / "registro.csv")
    monkeypatch.setattr(rutas, "RELEASES", tmp_path / "releases")
    return tmp_path


@pytest.fixture
def escribir_experimento():
    """escribir_experimento(carpeta, nombre, cabeza, vectores=["clip_b32__centro"]) -> ruta del .yaml"""
    def escribir(carpeta, nombre, cabeza, vectores=("clip_b32__centro",)):
        ruta = carpeta / f"{nombre}.yaml"
        ruta.write_text(
            f"nombre: {nombre}\npareja: base\nvectores: {json.dumps(list(vectores))}\n"
            f"cabeza: {json.dumps(cabeza)}\nentrenamiento: {{epocas: 30, paciencia: 5}}\n"
        )
        return ruta
    return escribir
