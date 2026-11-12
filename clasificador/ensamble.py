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
