"""Cierre: elegir un miembro por codificador con el dataset falso."""

import json

import pandas as pd


def test_mejores_por_codificador(entorno, escribir_experimento):
    from clasificador.cierre import mejores_por_codificador
    from clasificador.ensamble import ensamblar
    from clasificador.entrenar import cargar_experimento, entrenar
    from clasificador.evaluar import armar_registro, evaluar

    for nombre, cabeza, vectores in [("base", {"tipo": "lineal"}, ["clip_b32__centro"]),
                                     ("B_mlp", {"tipo": "mlp", "ocultas": [32]}, ["clip_b32__centro"]),
                                     ("A_otro", {"tipo": "lineal"}, ["otro__centro"])]:
        entrenar(cargar_experimento(escribir_experimento(entorno, nombre, cabeza, vectores)))
        evaluar(nombre, "val")
    ensamblar("C7_prueba", ["base", "A_otro"])
    evaluar("C7_prueba", "val")

    # uno evaluado con menos películas (como OCR con muestra): no puede ser miembro
    carpeta = entorno / "resultados"
    (carpeta / "A2_muestra").mkdir()
    metricas = json.loads((carpeta / "A_otro" / "metricas_val.json").read_text())
    metricas.update(nombre="A2_muestra", peliculas=10, F1_macro=0.99)
    (carpeta / "A2_muestra" / "metricas_val.json").write_text(json.dumps(metricas))
    fila = pd.read_csv(carpeta / "A_otro" / "registro_val.csv").assign(nombre="A2_muestra", F1_macro=0.99)
    fila.to_csv(carpeta / "A2_muestra" / "registro_val.csv", index=False)
    armar_registro()

    mejores = mejores_por_codificador()
    assert sorted(mejores["codificador"]) == ["clip_b32", "otro"]          # uno por codificador
    assert "C7_prueba" not in mejores.index and "A2_muestra" not in mejores.index
    assert list(mejores["F1_macro"]) == sorted(mejores["F1_macro"], reverse=True)
