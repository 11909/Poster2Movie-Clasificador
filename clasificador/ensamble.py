"""Ensamble promediando predicciones (Pareja C, idea 7b) — contrato 4.

Junta las cabezas de varios experimentos YA ENTRENADOS (los "miembros", normalmente uno por
codificador: base = CLIP B/32, A7_l14, A7_siglip), promedia sus 17 probabilidades (soft voting)
y recalibra un umbral por género sobre el promedio. Nada se entrena.

    python -m clasificador.ensamble C7_ensamble base A7_l14 A7_siglip
    python -m clasificador.evaluar C7_ensamble

Pasos:
1. Lee `<P2M_SALIDAS>/predicciones/<miembro>.npz` (`ids_val`, `prob_val`) y
   `<P2M_SALIDAS>/modelos/<miembro>.pt` de cada miembro.
2. `alinear`: se queda con las películas de validación que están en TODOS los miembros y pone
   las filas de cada uno en el mismo orden de ids.
3. Promedia las probabilidades (simple: pesos iguales).
4. Calibra los umbrales con `datos.calibrar_umbrales` sobre el promedio, con las etiquetas
   reales de validación de esos mismos ids. Los umbrales de cada miembro se descartan.
5. Guarda el combinado (`combinado.armar` + `combinado.guardar`) en
   `<P2M_SALIDAS>/modelos/<nombre>.pt`: las cabezas de los miembros, los pesos y los umbrales.

Pendiente: medir la memoria con el predictor en una T4 (celda del notebook C7); si no cabe
junto con la reseña y el cartel, publicar también una versión ligera (menos miembros).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from clasificador import combinado, rutas
from clasificador.datos import GENEROS, aplicar_umbrales, calibrar_umbrales
from clasificador.entrenar import commit_actual
from p2m import datos


def alinear(predicciones: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """{miembro: (ids_val, prob_val)} -> (ids comunes, [prob_val de cada miembro en ese orden]).

    Solo entran las películas que están en TODOS los miembros. Los ids comunes salen
    ordenados de menor a mayor y la lista de probabilidades sigue el orden del diccionario.
    """
    if not predicciones:
        raise ValueError("Se necesita al menos un miembro.")
    filas = {}  # miembro -> {id: fila}
    for nombre, (ids, prob) in predicciones.items():
        ids = np.asarray(ids, dtype=np.int64)
        if prob.ndim != 2 or len(ids) != len(prob):
            raise ValueError(f"{nombre}: se esperan {len(ids)} filas de probabilidades, hay {prob.shape}.")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{nombre}: hay ids repetidos.")
        filas[nombre] = {int(i): n for n, i in enumerate(ids)}

    comunes = set.intersection(*(set(f) for f in filas.values()))
    if not comunes:
        raise ValueError("Los miembros no tienen ninguna película en común.")
    ids_comunes = np.array(sorted(comunes), dtype=np.int64)
    probs = [np.asarray(prob)[[filas[nombre][i] for i in ids_comunes]]
             for nombre, (_, prob) in predicciones.items()]
    return ids_comunes, probs


def etiquetas_val(ids: np.ndarray) -> np.ndarray:
    """Matriz de géneros reales (n × 17) de esas películas de validación, en ese orden."""
    meta = datos.metadata("val").set_index("id")
    faltan = set(int(i) for i in ids) - set(meta.index)
    if faltan:
        raise ValueError(f"{len(faltan)} ids de las predicciones no están en validación.")
    return np.array([datos.vector_generos(list(meta.at[int(i), "generos"])) for i in ids], dtype=np.int64)


def cargar_predicciones(nombre: str) -> tuple[np.ndarray, np.ndarray]:
    ruta = rutas.carpeta_predicciones() / f"{nombre}.npz"
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro {ruta}: entrena antes el experimento '{nombre}'.")
    guardado = np.load(ruta)
    return guardado["ids_val"].astype(np.int64), guardado["prob_val"].astype(np.float32)


def ensamblar(nombre: str, miembros: list[str], pesos: list[float] | None = None) -> Path:
    """Arma y guarda el combinado `<P2M_SALIDAS>/modelos/<nombre>.pt` (contrato 4) con esos
    experimentos. Sin pesos: promedio simple."""
    if len(set(miembros)) != len(miembros):
        raise ValueError("Hay miembros repetidos.")
    if nombre in miembros:
        raise ValueError("El ensamble no puede llamarse igual que uno de sus miembros.")
    if pesos is not None and len(pesos) != len(miembros):
        raise ValueError("Debe haber un peso por miembro.")

    modelos = []
    for m in miembros:
        ruta = rutas.ruta_modelo(m)
        if not ruta.exists():
            raise FileNotFoundError(f"No encuentro {ruta}: entrena antes el experimento '{m}'.")
        guardado = torch.load(ruta, map_location="cpu", weights_only=False)
        if combinado.es_combinado(guardado):
            raise ValueError(f"'{m}' ya es un combinado: usa sus miembros directamente.")
        if list(guardado["generos"]) != list(GENEROS):
            raise ValueError(f"'{m}' tiene otros géneros u otro orden de géneros.")
        modelos.append(guardado)

    ids, probs = alinear({m: cargar_predicciones(m) for m in miembros})
    y = etiquetas_val(ids)

    w = np.ones(len(miembros)) if pesos is None else np.asarray(pesos, dtype=float)
    w = w / w.sum()
    promedio = sum(p * peso for p, peso in zip(probs, w))
    umbrales = calibrar_umbrales(y, promedio)

    print(f"Películas de validación en común: {len(ids)}")
    for m, p in zip(miembros, probs):
        f1 = f1_score(y, aplicar_umbrales(p, calibrar_umbrales(y, p)), average="macro", zero_division=0)
        print(f"  {m:<20} F1 macro en val = {f1:.4f}")
    f1 = f1_score(y, aplicar_umbrales(promedio, umbrales), average="macro", zero_division=0)
    print(f"  {nombre:<20} F1 macro en val = {f1:.4f}  (ensamble)")

    experimento = {
        "nombre": nombre, "pareja": "C",
        "descripcion": f"Idea 7b: ensamble (soft voting) de {', '.join(miembros)}.",
        "miembros": list(miembros),
    }
    ruta = combinado.guardar(
        combinado.armar(modelos, umbrales, pesos=w.tolist(), experimento=experimento, commit=commit_actual()),
        rutas.ruta_modelo(nombre),
    )
    print(f"Guardado en {ruta}")
    return ruta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nombre", help="nombre del ensamble, p. ej. C7_ensamble")
    parser.add_argument("miembros", nargs="+", help="experimentos ya entrenados, p. ej. base A7_l14 A7_siglip")
    parser.add_argument("--pesos", nargs="+", type=float, default=None,
                        help="uno por miembro (se normalizan). Sin pesos: promedio simple")
    args = parser.parse_args()
    ensamblar(args.nombre, args.miembros, args.pesos)
