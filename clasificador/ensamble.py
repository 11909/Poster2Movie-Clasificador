"""Ensamble promediando predicciones — PENDIENTE (Pareja C, idea 7b).

El FORMATO del .pt ya está resuelto (contrato 4, `clasificador/combinado.py`): el predictor
y `evaluar` ya saben cargarlo. Lo que falta es DECIDIR qué se combina:

1. Elegir los miembros: un experimento por codificador (B/32, L/14, SigLIP...), ya entrenados.
2. Leer `<P2M_SALIDAS>/predicciones/<nombre>.npz` de cada uno (`ids_val`, `prob_val`) y
   alinearlos por id (no todos tienen por qué tener las mismas películas).
3. Promediar `prob_val`: simple, o con pesos elegidos en val (p. ej. rejilla de pesos).
4. Recalibrar los umbrales con `clasificador.datos.calibrar_umbrales` sobre el promedio.
5. Guardar:
       combinado.guardar(combinado.armar([torch.load(pt) ...], umbrales, pesos=pesos,
                                         experimento={"nombre": "C_ensamble", "pareja": "C", ...}),
                         rutas.ruta_modelo("C_ensamble"))
6. `python -m clasificador.evaluar C_ensamble` y compararlo contra cada miembro solo.
7. Medir la memoria con el predictor en una T4 (`Clasificador(ruta)`); si no cabe junto con
   la reseña y el cartel, publicar también una versión ligera.
"""

import argparse

import numpy as np


def alinear(predicciones: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """{miembro: (ids_val, prob_val)} -> (ids comunes, [prob_val de cada miembro en ese orden])."""
    raise NotImplementedError("Ensamble: pendiente de la Pareja C (idea 7b).")


def ensamblar(nombre: str, miembros: list[str], pesos: list[float] | None = None) -> None:
    """Arma y guarda el combinado `<P2M_SALIDAS>/modelos/<nombre>.pt` (contrato 4) con esos
    experimentos. Sin pesos: elegirlos en val (o promedio simple)."""
    raise NotImplementedError("Ensamble: pendiente de la Pareja C (idea 7b).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nombre", help="nombre del ensamble, p. ej. C7_ensamble")
    parser.add_argument("miembros", nargs="+", help="experimentos ya entrenados, p. ej. base A7_l14 A7_siglip")
    parser.add_argument("--pesos", nargs="+", type=float, default=None)
    args = parser.parse_args()
    ensamblar(args.nombre, args.miembros, args.pesos)
