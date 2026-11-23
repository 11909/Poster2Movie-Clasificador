"""Ordena los experimentos de una idea contra la base, con la regla de publicación del README.

    python -m clasificador.ranking B3_                 # los B3_* en validación, mejores primero
    python -m clasificador.ranking B4_ --top 5

Regla 6 del README: un modelo es candidato a release si su F1 macro en validación no baja respecto
a la base Y el promedio del F1 de los 4 géneros débiles sube. Aquí eso es la columna `cumple`.
Se ordena por `cumple` y luego por el promedio de los débiles (el objetivo del equipo), con el F1
macro como desempate. Solo mira validación: test no se usa para elegir.

Usa la ÚLTIMA evaluación de cada experimento (si se evaluó varias veces, cuenta la más reciente).
"""

import argparse

import pandas as pd

from clasificador.datos import GENEROS_DEBILES
from clasificador.evaluar import armar_registro

COLUMNAS_DEBILES = [f"F1_{g}" for g in GENEROS_DEBILES]


def ranking(prefijo: str, base: str = "base", split: str = "val") -> pd.DataFrame:
    registro = armar_registro()
    registro = registro[registro["split"] == split]
    ultimas = registro.drop_duplicates("nombre", keep="last").set_index("nombre")
    if base not in ultimas.index:
        raise FileNotFoundError(f"No hay evaluación de '{base}' en {split}: evalúenla primero "
                                f"(`python -m clasificador.evaluar {base}`).")
    numeros = ultimas[["F1_macro", "F1_micro", "mAP", *COLUMNAS_DEBILES]].astype(float)
    numeros["debiles_prom"] = numeros[COLUMNAS_DEBILES].mean(axis=1)
    referencia = numeros.loc[base]

    tabla = numeros[numeros.index.str.startswith(prefijo)].copy()
    tabla["d_macro"] = tabla["F1_macro"] - referencia["F1_macro"]
    tabla["d_debiles"] = tabla["debiles_prom"] - referencia["debiles_prom"]
    tabla["cumple"] = (tabla["d_macro"] >= 0) & (tabla["d_debiles"] > 0)
    return tabla.sort_values(["cumple", "debiles_prom", "F1_macro"], ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefijo", help="p. ej. B3_ (todos los experimentos cuyo nombre empieza así)")
    parser.add_argument("--base", default="base")
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    resultado = ranking(args.prefijo, args.base, args.split)
    columnas = ["F1_macro", "d_macro", "debiles_prom", "d_debiles", "mAP", "cumple"]
    print(resultado[columnas].head(args.top).to_string(
        float_format=lambda v: f"{v:+.3f}" if abs(v) < 0.1 and v != 0 else f"{v:.3f}"))
    print(f"\n{int(resultado['cumple'].sum())} de {len(resultado)} cumplen la regla "
          f"(macro no baja y los 4 débiles suben) contra {args.base}.")
