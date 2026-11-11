"""Ensamble promediando predicciones — PENDIENTE (Pareja C, idea 7b).

Qué va aquí:
1. Leer `<P2M_SALIDAS>/predicciones/<nombre>.npz` de cada experimento (uno por codificador).
2. Promediar `prob_val` (simple o con pesos elegidos en val) y recalibrar los umbrales
   con `clasificador.datos.calibrar_umbrales`.
3. Guardar un .pt que el predictor sepa reconstruir (varios codificadores + varias cabezas).
   Revisar que quepa en la memoria de la demo (GPU T4 junto con los otros módulos).
"""
