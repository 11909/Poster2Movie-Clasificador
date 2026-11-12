"""Apilamiento (stacking) — PENDIENTE (Pareja C, idea 6).

El FORMATO ya está resuelto (contrato 4, `clasificador/combinado.py`): la etapa 2 va en la
clave `apilamiento` de un combinado y el predictor y `evaluar` ya la aplican. Lo que falta:

1. Predicciones de train FUERA DE PLIEGUE: partir train en 5 pedazos (por franquicia,
   con la columna `coleccion` de la metadata, para que una saga no quede en dos pedazos),
   entrenar la etapa 1 cinco veces y predecir cada pedazo con el modelo que no lo vio.
2. Etapa 2: una cabeza 17 -> 17 (`cabezas.construir_cabeza({"tipo": "lineal"}, 17, 17)`;
   después, si ayuda, un MLP). Con `residual: True` su salida se SUMA al logit de la
   probabilidad original (solo ajusta). Regularizarla. Se entrena con las de train y se
   calibran los umbrales en val, sobre la salida final.
3. Guardar:
       combinado.armar(miembros, umbrales, pesos=..., apilamiento={"arquitectura": {"tipo": "lineal"},
                       "capa": etapa2.state_dict(), "residual": True}, experimento={...})
4. Guardar la matriz de pesos 17 × 17 como mapa de calor para el reporte.
"""
