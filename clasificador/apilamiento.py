"""Apilamiento (stacking) — PENDIENTE (Pareja C, idea 6).

Qué va aquí:
1. Predicciones de train FUERA DE PLIEGUE: partir train en 5 pedazos (por franquicia,
   con la columna `coleccion` de la metadata, para que una saga no quede en dos pedazos),
   entrenar la etapa 1 cinco veces y predecir cada pedazo con el modelo que no lo vio.
2. Etapa 2: una capa lineal 17 -> 17 (después, si ayuda, un MLP) que recibe esas
   probabilidades y da las corregidas. Se entrena con las de train y se calibra en val.
3. Guardar la matriz de pesos 17 × 17 como mapa de calor para el reporte.
"""
