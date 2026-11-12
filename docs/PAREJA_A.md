# Pareja A — Qué ve el modelo

[← README](../README.md) · [Contratos](CONTRATOS.md) · [Pareja B](PAREJA_B.md) · [Pareja C](PAREJA_C.md)

**Objetivo:** darle al modelo más información del cartel: verlo completo, leer su texto y sacar
vectores con codificadores más potentes. **Producen vectores** (contrato 1) que usan B y C.

**Tu archivo:** `clasificador/codificadores.py`. La misma clase que calcula los vectores es la que
usa el predictor en la demo: el cartel se procesa igual al entrenar y al predecir.

| Idea | Rama | Notebook | GPU |
| --- | --- | --- | --- |
| 1 · Multi-recorte | `A-multirecorte` | `notebooks/A1_multirecorte.ipynb` | sí |
| 2 · OCR | `A-ocr` | `notebooks/A2_ocr.ipynb` | sí |
| 7a · Nuevos codificadores | `A-codificadores` | `notebooks/A7_codificadores.ipynb` | sí |

**Orden sugerido:** 7a primero (C la necesita para el ensamble), luego 1, luego 2.

---

## Idea 7a · Nuevos codificadores

CLIP L/14 (`openai/clip-vit-large-patch14`, 768) y SigLIP base (`google/siglip-base-patch16-224`,
768). Opcionales: SigLIP so400m (1152) y DINOv2 (`facebook/dinov2-base`, 768, sin texto).

- [ ] `clip_l14`: moverlo de `CODIFICADORES_PENDIENTES` a `CODIFICADORES` (usa la misma clase `CodificadorCLIP`).
- [ ] SigLIP: nueva clase `CodificadorSigLIP(Codificador)` con `AutoModel`/`AutoProcessor` (no tiene
      `projection_dim`: `dim` sale del tamaño del vector) y registrarla en `crear_codificador`.
- [ ] Vectores de recorte central: `python -m clasificador.codificadores clip_l14` y `... siglip_base`.
- [ ] Una cabeza por codificador: `experimentos/A7_l14.yaml`, `A7_siglip.yaml` → entrenar, evaluar, comparar.
- [ ] Quitar el `xfail` de `test_idea7a_codificadores_nuevos`.

**Listo cuando** los vectores están en `poster2movie_clasificador/vectores/` para B y C, y los
dos experimentos en el registro.

## Idea 1 · Ver el cartel completo

Hoy CLIP recorta el cuadrado del centro y tira arriba y abajo (título, frase promocional).

- [ ] `recortes(imagen, variante)`: `relleno` → 1 imagen cuadrada con bordes (cartel completo);
      `3recortes` → 3 cuadrados (arriba, centro, abajo).
- [ ] `CodificadorCLIP.codificar` usa `recortes()` para esas variantes. Con 3 recortes: promediar
      (dim 512) o concatenar (dim 1536); decidan y documenten.
- [ ] Mover `relleno` y `3recortes` de `VARIANTES_PENDIENTES` a `VARIANTES`.
- [ ] Vectores: `python -m clasificador.codificadores clip_b32 --variante relleno` (y `3recortes`).
- [ ] Experimentos `A1_relleno`, `A1_3recortes`. Quitar el `xfail` de `test_idea1_relleno_y_tres_recortes`.

**Listo cuando** su resultado en validación está en el registro. Si una variante ayuda, sáquenla
también para el mejor codificador de 7a.

## Idea 2 · Leer el texto del cartel (OCR)

- [ ] Variante `ocr`: EasyOCR lee el texto → `codificar_texto` de CLIP (máx. 77 tokens) → vector de
      512. Sin texto: vector de ceros (u otra decisión, documentada). Agregar `easyocr` a `requirements.txt`.
- [ ] Prueba chica: `python -m clasificador.codificadores clip_b32 --variante ocr --muestra 2000`
      (2 mil películas fijas; el resultado es ruidoso, sirve para decidir).
- [ ] Experimento `A2_ocr` (`vectores: [clip_b32__centro, clip_b32__ocr]`).
- [ ] Si ayuda: todo el dataset (varias horas de GPU; repartan entre cuentas).
- [ ] Quitar el `xfail` de `test_idea2_ocr_registrado`.

**Listo cuando** está probado con 2 mil carteles y, si ayudó, corrido en todo el dataset.

---

## Cuidados

- **Cuota de GPU de Colab:** calculen primero lo prioritario y repartan entre sus dos cuentas. El
  cálculo guarda por partes: si Colab se desconecta, al volver a correrlo continúa.
- **`codificadores.py` también lo toca la Pareja B** (variante `zeroshot`). Cambios chicos y
  localizados; PRs cortos para no chocar.
- **Para la demo:** todo codificador o variante que termine en un release hay que copiarlo a
  `p2m/codificadores.py` del repo principal (ver [Pareja C](PAREJA_C.md#integración-con-el-repo-principal)).
