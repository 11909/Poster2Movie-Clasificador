"""Cierre: de todos los experimentos ya evaluados, elige los miembros del modelo final (Pareja C).

    python -m clasificador.cierre            # el mejor experimento de cada codificador + la tabla final

Regla: por cada codificador (CLIP B/32, CLIP L/14, SigLIP...) se queda con UN experimento, el primero
según `ranking` (cumple la regla de publicación > promedio de los 4 débiles > F1 macro). Esos son
los miembros para el ensamble (7b) y el apilamiento (6), ordenados por F1 macro.

Se excluyen:
- los combinados (`C*`): un combinado no puede ser miembro de otro;
- los que se evaluaron con MENOS películas de validación que la base (p. ej. OCR con muestra de 2 mil):
  el ensamble se queda solo con las películas que están en todos los miembros, y uno así encogería la
  validación de todos.

Solo mira validación: test no se usa para elegir.
"""

import argparse
import json

import pandas as pd

from clasificador import rutas
from clasificador.evaluar import armar_registro
from clasificador.ranking import ranking


def peliculas_val(nombre: str) -> int | None:
    ruta = rutas.RESULTADOS / nombre / "metricas_val.json"
    return json.loads(ruta.read_text(encoding="utf-8"))["peliculas"] if ruta.exists() else None


def tabla_completa(base: str = "base") -> pd.DataFrame:
    """Todos los experimentos evaluados en val contra la base, con su codificador y películas."""
    tabla = ranking("", base)
    vectores = armar_registro().drop_duplicates("nombre", keep="last").set_index("nombre")["vectores"]
    tabla["vectores"] = vectores.reindex(tabla.index).fillna("")
    tabla["codificador"] = tabla["vectores"].str.split("+").str[0].str.split("__").str[0]
    tabla["peliculas"] = [peliculas_val(n) for n in tabla.index]
    return tabla


def mejores_por_codificador(base: str = "base") -> pd.DataFrame:
    """Un experimento por codificador (el primero según `ranking`), ordenados por F1 macro."""
    tabla = tabla_completa(base)
    completas = tabla.loc[base, "peliculas"]
    candidatos = tabla[~tabla.index.str.startswith("C") & (tabla["peliculas"] == completas)]
    return candidatos.groupby("codificador", sort=False).head(1).sort_values("F1_macro", ascending=False)


COLUMNAS = ["codificador", "F1_macro", "d_macro", "debiles_prom", "d_debiles", "mAP", "cumple"]


def formato(v):
    return f"{v:+.3f}" if isinstance(v, float) and abs(v) < 0.1 and v != 0 else (f"{v:.3f}" if isinstance(v, float) else v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="base")
    args = parser.parse_args()
    mejores = mejores_por_codificador(args.base)
    print("Miembros (el mejor de cada codificador, por F1 macro):")
    print(mejores[COLUMNAS].to_string(float_format=formato))
    print("\nMIEMBROS =", list(mejores.index))
    combinados = tabla_completa(args.base)
    combinados = combinados[combinados.index.str.startswith("C")]
    if len(combinados):
        print("\nCombinados (ensamble / apilamiento) contra la base:")
        print(combinados[COLUMNAS[1:]].to_string(float_format=formato))
