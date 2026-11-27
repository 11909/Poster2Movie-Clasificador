# Pareja C — Combinar y publicar

[← README](../README.md) · [Contratos](CONTRATOS.md) · [Pareja A](PAREJA_A.md) · [Pareja B](PAREJA_B.md)

**Objetivo:** juntar lo que hacen A y B en un solo modelo mejor, publicar cada versión como
release y llevar el modelo final al repo principal.

**El formato ya está resuelto:** el `.pt` combinado (contrato 4, `clasificador/combinado.py`) ya
lo cargan el predictor y `evaluar`. Lo que falta es **decidir qué combinar** y entrenar la etapa 2.

| Tarea | Rama | Notebook | Archivo | GPU |
| --- | --- | --- | --- | --- |
| 6 · Apilamiento | `C-apilamiento` | `notebooks/C6_apilamiento.ipynb` | `clasificador/apilamiento.py` | no |
| 7b · Ensamble | `C-ensamble` | `notebooks/C7_ensamble.ipynb` | `clasificador/ensamble.py` | para medir memoria |
| Releases | `release-<tag>` | `notebooks/C_release.ipynb` | `clasificador/release.py` (ya existe) | no |

**Primero:** `notebooks/00_reproducir_base.ipynb` (una sola vez para todo el equipo) → `v0.1.0`.
En test debe dar F1 macro **0.537**; si no, no se publica y se avisa.

---

## Idea 7b · Ensamble

Una cabeza por codificador (B/32, L/14, SigLIP), promediar sus 17 probabilidades y recalibrar
umbrales. Nada se entrena desde cero.

- [x] `alinear`: junta las `predicciones/<nombre>.npz` (`ids_val`, `prob_val`) de cada miembro por id.
- [x] `ensamblar`: promedio simple (o `--pesos` a mano), umbrales con `calibrar_umbrales`, y
      guardar con `combinado.guardar(combinado.armar(...))`. Pesos elegidos en val: por ahora no.
- [ ] `python -m clasificador.ensamble C7_ensamble base A7_l14 A7_siglip` → evaluar → comparar
      contra cada miembro solo. Mientras A termina 7a, prueben con base + otros experimentos B/32.
- [ ] Medir el pico de memoria en una T4 (celda del notebook). Si no cabe junto con la reseña y el
      cartel, publicar también una versión ligera (menos miembros).
- [x] Quitar el `xfail` de `test_idea7b_alinear_predicciones` (y prueba del flujo en `tests/test_ensamble.py`).

**Listo cuando** el ensamble está en validación contra cada modelo solo.

## Idea 6 · Apilamiento (stacking)

Una etapa 2 corrige las 17 probabilidades con cómo se juntan los géneros (Misterio casi siempre
va con Thriller o Crimen).

- [x] `pliegues`: 5 pliegues de train **por franquicia** (columna `coleccion`; una saga nunca en dos).
      Las franquicias se barajan con semilla fija y se reparten de la más grande a la más chica
      al pliegue con menos películas: tamaños casi iguales y no depende del orden de las filas.
- [x] `apilar`: predicciones **fuera de pliegue** de la etapa 1 (cada miembro se reentrena 5 veces
      con su misma receta, `entrenar.entrenar_cabeza`, y predice el pliegue que no vio), etapa 2
      lineal 17 → 17 con `residual: True` que arranca en ceros (solo ajusta), regularizada con L2
      hacia cero (`--l2`) y con paro temprano por la pérdida en val; umbrales calibrados en val;
      guardado como combinado. MLP disponible con `--etapa2 mlp`. Acepta varios miembros
      (apilar sobre un ensamble) con `--pesos` como el ensamble.
- [ ] `python -m clasificador.apilamiento C6_apilamiento base` → evaluar → comparar (en Colab, con los datos reales).
- [x] Mapa de calor 17 × 17: `apilamiento.mapa_calor` (lo corre el comando y la celda del notebook)
      a partir de `resultados/<nombre>/pesos_etapa2.csv`. Falta generarlo con los datos reales.
- [x] Quitar el `xfail` de `test_idea6_pliegues_por_franquicia` (y flujo completo en `tests/test_apilamiento.py`).

**Listo cuando** está en validación y el mapa 17 × 17 queda en `resultados/C6_apilamiento/`.

---

## Releases

Se publica cuando **el F1 macro en validación no baja y el promedio de los 4 débiles sube** contra
el último release. Versiones: PARCHE `v0.2.1` (arreglo), MENOR `v0.3.0` (modelo que mejora),
MAYOR `v1.0.0` (el que se entrega). Un release no se reescribe.

1. Rama `release-<tag>`: `python -m clasificador.release <tag> <experimento> --notas "..."` (mide val
   y test, copia `.pt`, métricas, receta y `LEEME.md` a `releases/<tag>/`).
2. Pull Request → al fusionarlo, en `main`: `git tag <tag> && git push origin <tag>`.
3. GitHub Actions publica el release. Todo esto lo hace el notebook `C_release`.

## Integración con el repo principal

El repo principal (`AndreaHortaHernandez/poster2movie`) usa nuestros releases **sin instalar este
repo**:

- `p2m/clasificador.py` tiene `RELEASE_CLASIFICADOR = {"tag": ..., "archivo": ..., "sha256": ...}`.
  `Clasificador()` descarga ese `.pt` del release (con caché); sin red, usa el base. La demo y
  `evaluar_resena --generos predichos` lo usan; el juez de `evaluar_cartel` sigue siendo el base.
- `p2m/codificadores.py` es una **copia** de la parte de inferencia de nuestros codificadores, y
  carga `.pt` normales y combinados (contrato 4).

Para que la demo use un release nuevo: PR al repo principal que (1) fije `RELEASE_CLASIFICADOR`,
(2) copie a `p2m/codificadores.py` los codificadores o variantes nuevos y a `construir_cabeza` las
cabezas nuevas, y (3) compruebe que `p2m.clasificador.Clasificador(ruta)` da las mismas
probabilidades que `clasificador.predictor.Clasificador(ruta)`. Hagan una integración de prueba
temprana con `v0.1.0`, no esperen a la entrega.
