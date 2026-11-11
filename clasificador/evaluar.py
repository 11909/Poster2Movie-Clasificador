"""Evalúa un modelo entrenado (contrato 3).

    python -m clasificador.evaluar base                 # validación (lo normal)
    python -m clasificador.evaluar base --split test --para-release

Escribe en `resultados/<nombre>/`:
- `metricas_<split>.json`: F1 macro, F1 micro, F1 por película, mAP, acierto exacto,
  y la línea base trivial (siempre los 2 géneros más comunes).
- `por_genero_<split>.csv`: precisión, recall, F1 y umbral de cada género.

Y agrega una fila a `resultados/registro.csv`.

Test solo se mide al publicar un release (`python -m clasificador.release`). Si elegimos
modelos mirando test, los números finales salen inflados.
"""

import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support

from clasificador import rutas
from clasificador.cabezas import cargar_cabeza
from clasificador.datos import GENEROS, GENEROS_DEBILES, aplicar_umbrales, cargar_split
from clasificador.entrenar import commit_actual
from p2m import config

COLUMNAS_REGISTRO = ["fecha", "nombre", "pareja", "split", "F1_macro", "F1_micro", "mAP",
                     *[f"F1_{g}" for g in GENEROS_DEBILES], "vectores", "commit"]


def metricas(y: np.ndarray, prediccion: np.ndarray, probabilidades: np.ndarray | None) -> dict:
    return {
        "F1_macro": float(f1_score(y, prediccion, average="macro", zero_division=0)),
        "F1_micro": float(f1_score(y, prediccion, average="micro", zero_division=0)),
        "F1_por_pelicula": float(f1_score(y, prediccion, average="samples", zero_division=0)),
        "mAP": float(average_precision_score(y, probabilidades, average="macro")) if probabilidades is not None else None,
        "acierto_exacto": float(accuracy_score(y, prediccion)),
    }


def evaluar(nombre_o_ruta: str, split: str = "val", para_release: bool = False) -> dict:
    if split == "test" and not para_release:
        raise SystemExit("Test solo se mide al publicar un release: usa `python -m clasificador.release`. "
                         "Para elegir modelos, evalúa en val.")
    guardado = torch.load(rutas.ruta_modelo(nombre_o_ruta), map_location="cpu", weights_only=False)
    experimento = guardado.get("experimento") or {}
    nombre = experimento.get("nombre", str(nombre_o_ruta))
    vectores = guardado.get("vectores") or ["clip_b32__centro"]  # un .pt del repo principal

    x, y, _ = cargar_split(split, vectores)
    cabeza = cargar_cabeza(guardado, x.shape[1])
    with torch.no_grad():
        prob = torch.sigmoid(cabeza(torch.tensor(x))).numpy()
    prediccion = aplicar_umbrales(prob, np.asarray(guardado["umbrales"]))

    _, y_train, _ = cargar_split("train", vectores)
    comunes = np.argsort(-y_train.sum(axis=0))[:2]
    trivial = np.zeros_like(y)
    trivial[:, comunes] = 1

    precision, recall, f1, soporte = precision_recall_fscore_support(y, prediccion, zero_division=0)
    por_genero = pd.DataFrame({
        "genero": GENEROS, "genero_es": [config.GENEROS_ES[g] for g in GENEROS],
        "peliculas": soporte, "precision": precision, "recall": recall, "F1": f1,
        "umbral": np.asarray(guardado["umbrales"]),
    })
    resultado = {
        "nombre": nombre, "pareja": experimento.get("pareja", ""), "split": split, "peliculas": int(len(y)),
        "vectores": vectores, "commit_modelo": guardado.get("commit", ""),
        **metricas(y, prediccion, prob),
        "F1_debiles": {g: float(f1[GENEROS.index(g)]) for g in GENEROS_DEBILES},
        "trivial": {"generos": [GENEROS[i] for i in comunes], **metricas(y, trivial, None)},
    }

    carpeta = rutas.RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"metricas_{split}.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    por_genero.to_csv(carpeta / f"por_genero_{split}.csv", index=False, float_format="%.4f")
    registrar(resultado)

    print(f"{nombre} en {split} ({len(y)} películas): F1 macro {resultado['F1_macro']:.3f} | "
          f"F1 micro {resultado['F1_micro']:.3f} | mAP {resultado['mAP']:.3f} | "
          f"trivial {resultado['trivial']['F1_macro']:.3f}")
    print(por_genero.sort_values("F1", ascending=False).to_string(index=False, float_format="{:.3f}".format))
    return resultado


def registrar(resultado: dict) -> None:
    rutas.REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not rutas.REGISTRO.exists() or rutas.REGISTRO.stat().st_size == 0
    with open(rutas.REGISTRO, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_REGISTRO)
        if nuevo:
            escritor.writeheader()
        escritor.writerow({
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nombre": resultado["nombre"], "pareja": resultado["pareja"], "split": resultado["split"],
            "F1_macro": f"{resultado['F1_macro']:.4f}", "F1_micro": f"{resultado['F1_micro']:.4f}",
            "mAP": f"{resultado['mAP']:.4f}",
            **{f"F1_{g}": f"{v:.4f}" for g, v in resultado["F1_debiles"].items()},
            "vectores": "+".join(resultado["vectores"]), "commit": commit_actual(),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modelo", help="nombre del experimento (busca <P2M_SALIDAS>/modelos/<nombre>.pt) o ruta a un .pt")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--para-release", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    evaluar(args.modelo, args.split, args.para_release)
