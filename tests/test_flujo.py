"""El flujo completo con un dataset falso: entrenar -> evaluar -> comparar -> release.

No necesita GPU, ni Drive, ni descargar modelos.
"""

import pandas as pd
import pytest

from p2m import config

N = 600


def test_flujo_completo(entorno, escribir_experimento):
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
