"""Compara dos modelos género por género.

    python -m clasificador.comparar base B_mlp_512           # en validación
    python -m clasificador.comparar v0.1.0 v0.2.0 --split test

Cada argumento es un experimento (`resultados/<nombre>/`) o un release (`releases/<tag>/`).
Los cuatro géneros débiles salen primero.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from clasificador import rutas
from clasificador.datos import GENEROS_DEBILES


def carpeta_de(nombre: str) -> Path:
    for carpeta in (rutas.RESULTADOS / nombre, rutas.RELEASES / nombre):
        if carpeta.is_dir():
            return carpeta
    raise FileNotFoundError(f"No hay resultados de '{nombre}' en resultados/ ni en releases/.")


def comparar(a: str, b: str, split: str = "val") -> pd.DataFrame:
    tablas, generales = [], []
    for nombre in (a, b):
        carpeta = carpeta_de(nombre)
        archivo = carpeta / f"por_genero_{split}.csv"
        if not archivo.exists():
            raise FileNotFoundError(f"'{nombre}' no tiene resultados en {split}: falta {archivo}.")
        tablas.append(pd.read_csv(archivo).set_index("genero"))
        generales.append(json.loads((carpeta / f"metricas_{split}.json").read_text(encoding="utf-8")))

    tabla = pd.DataFrame({
        "genero": tablas[0]["genero_es"],
        a: tablas[0]["F1"], b: tablas[1]["F1"],
    })
    tabla["diferencia"] = tabla[b] - tabla[a]
    tabla["debil"] = tabla.index.isin(GENEROS_DEBILES)
    tabla = tabla.sort_values(["debil", a], ascending=[False, True])

    print(f"Split: {split}")
    for clave in ("F1_macro", "F1_micro", "mAP"):
        va, vb = generales[0][clave], generales[1][clave]
        print(f"  {clave:9s} {a}: {va:.3f}   {b}: {vb:.3f}   diferencia: {vb - va:+.3f}")
    print()
    formato = {a: "{:.3f}".format, b: "{:.3f}".format, "diferencia": "{:+.3f}".format}
    print(tabla.drop(columns="debil").to_string(index=False, formatters=formato))
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()
    comparar(args.a, args.b, args.split)
