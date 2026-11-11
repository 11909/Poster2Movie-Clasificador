"""Entrena un experimento descrito en `experimentos/<nombre>.yaml` (contrato 2).

    python -m clasificador.entrenar experimentos/base.yaml

Sale:
- `<P2M_SALIDAS>/modelos/<nombre>.pt`: la cabeza, los umbrales y todo lo necesario
  para reconstruirla (mismas claves que el .pt del repo principal, más
  `vectores`, `experimento` y `commit`).
- `<P2M_SALIDAS>/predicciones/<nombre>.npz`: ids y probabilidades de validación
  (`ids_val`, `prob_val`), para el ensamble y el apilamiento.

El entrenamiento es el de la base: AdamW, lotes al azar, se queda con la época de
mejor F1 macro en validación (umbral 0.5) y al final calibra un umbral por género.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from clasificador import RAIZ, rutas
from clasificador.cabezas import construir_cabeza
from clasificador.datos import GENEROS, aplicar_umbrales, calibrar_umbrales, cargar_split
from clasificador.perdidas import construir_perdida

ENTRENAMIENTO_BASE = {"epocas": 200, "tasa": 1e-3, "paciencia": 10, "lote": 256,
                      "weight_decay": 1e-4, "semilla": 42}


def commit_actual() -> str:
    """Commit del repo; con `+cambios` si hay cambios sin guardar (el experimento no es reproducible)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                                capture_output=True, text=True, check=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain", "--", "clasificador", "experimentos"],
                               cwd=RAIZ, capture_output=True, text=True).stdout.strip()
        return commit + ("+cambios" if sucio else "")
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def cargar_experimento(ruta: str | Path) -> dict:
    ruta = Path(ruta)
    experimento = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    for clave in ("nombre", "pareja", "vectores"):
        if clave not in experimento:
            raise ValueError(f"{ruta}: falta la clave '{clave}'.")
    if experimento["nombre"] != ruta.stem:
        raise ValueError(f"{ruta}: 'nombre' ({experimento['nombre']}) debe ser igual al nombre del archivo ({ruta.stem}).")
    if isinstance(experimento["vectores"], str):
        experimento["vectores"] = [experimento["vectores"]]
    experimento.setdefault("cabeza", {"tipo": "lineal"})
    experimento.setdefault("perdida", {"tipo": "bce"})
    experimento["entrenamiento"] = {**ENTRENAMIENTO_BASE, **(experimento.get("entrenamiento") or {})}
    return experimento


def nombre_embeddings(vectores: list[str]) -> str | None:
    """Clave `embeddings` para que el evaluador del repo principal también encuentre los vectores."""
    if len(vectores) != 1:
        return None
    return rutas.ALIAS_VECTORES.get(vectores[0], f"vectores/{vectores[0]}.npz")


def entrenar(experimento: dict) -> Path:
    e = experimento["entrenamiento"]
    x_train, y_train, _ = cargar_split("train", experimento["vectores"])
    x_val, y_val, ids_val = cargar_split("val", experimento["vectores"])
    print(f"{experimento['nombre']}: train {len(x_train)} | val {len(x_val)} | entrada {x_train.shape[1]}")

    torch.manual_seed(e["semilla"])
    cabeza = construir_cabeza(experimento["cabeza"], x_train.shape[1], len(GENEROS))
    perdida = construir_perdida(experimento["perdida"], y_train)
    optimizador = torch.optim.AdamW(cabeza.parameters(), lr=e["tasa"], weight_decay=e["weight_decay"])

    xt, yt = torch.tensor(x_train), torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(x_val)
    mejor, sin_mejorar, mejor_estado, epoca = -1.0, 0, None, 0
    for epoca in range(1, e["epocas"] + 1):
        cabeza.train()
        for i in torch.randperm(len(xt)).split(e["lote"]):
            optimizador.zero_grad()
            perdida(cabeza(xt[i]), yt[i]).backward()
            optimizador.step()
        cabeza.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(cabeza(xv)).numpy()
        f1_macro = f1_score(y_val, aplicar_umbrales(prob_val, np.full(len(GENEROS), 0.5)),
                            average="macro", zero_division=0)
        if f1_macro > mejor:
            mejor, sin_mejorar = f1_macro, 0
            mejor_estado = {k: v.clone() for k, v in cabeza.state_dict().items()}
        else:
            sin_mejorar += 1
            if sin_mejorar >= e["paciencia"]:
                break
    print(f"Época {epoca}: mejor F1 macro en val (umbral 0.5) = {mejor:.4f}")

    cabeza.load_state_dict(mejor_estado)
    cabeza.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(cabeza(xv)).numpy()
    umbrales = calibrar_umbrales(y_val, prob_val)
    f1_calibrado = f1_score(y_val, aplicar_umbrales(prob_val, umbrales), average="macro", zero_division=0)
    print(f"Con umbrales por género: F1 macro en val = {f1_calibrado:.4f}")

    vectores = experimento["vectores"]
    codificador = vectores[0].partition("__")[0] if len(vectores) == 1 else None
    from clasificador.codificadores import CODIFICADORES  # sin cargar transformers

    ruta = rutas.carpeta_modelos() / f"{experimento['nombre']}.pt"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "modelo_clip": CODIFICADORES.get(codificador), "embeddings": nombre_embeddings(vectores),
        "arquitectura": experimento["cabeza"], "generos": GENEROS, "capa": mejor_estado, "umbrales": umbrales,
        "vectores": vectores, "experimento": experimento, "commit": commit_actual(),
    }, ruta)
    predicciones = rutas.carpeta_predicciones() / f"{experimento['nombre']}.npz"
    predicciones.parent.mkdir(parents=True, exist_ok=True)
    np.savez(predicciones, ids_val=np.array(ids_val), prob_val=prob_val.astype(np.float32))
    print(f"Guardado en {ruta}")
    return ruta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experimento", help="ruta al .yaml, p. ej. experimentos/base.yaml")
    args = parser.parse_args()
    entrenar(cargar_experimento(args.experimento))
