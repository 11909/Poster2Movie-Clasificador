# Pareja B — Cómo aprende el modelo

[← README](../README.md) · [Contratos](CONTRATOS.md) · [Pareja A](PAREJA_A.md) · [Pareja C](PAREJA_C.md)

**Objetivo:** mejorar la cabeza que entrenamos: más capas, una pérdida que se enfoque en los
casos difíciles y pistas de texto de CLIP como entradas extra.

**Empiezan desde el día uno**, sin esperar a la Pareja A: con los vectores `clip_b32__centro`
que ya existen. Entrenar una cabeza tarda segundos y no necesita GPU.

| Idea | Rama | Notebook | Archivo | GPU |
| --- | --- | --- | --- | --- |
| 3 · Cabeza MLP | `B-cabezas` | `notebooks/B3_cabezas.ipynb` | `clasificador/cabezas.py` | no |
| 4 · Focal Loss | `B-focal` | `notebooks/B4_focal.ipynb` | `clasificador/perdidas.py` | no |
| 5 · Zero-shot | `B-zeroshot` | `notebooks/B5_zeroshot.ipynb` | `clasificador/zeroshot.py` | una vez |

---

## Idea 3 · Cabeza con capas ocultas (MLP)

Ya existe `{"tipo": "mlp", "ocultas": [...], "dropout": ...}` con ReLU. Una clave desconocida da
error (no se ignora).

- [ ] En `cabezas.py`: `activacion: gelu` y `layernorm: true` (hoy lanzan `NotImplementedError`).
- [ ] Rejilla desde el notebook: 1 a 3 capas ocultas, ancho 256/512/1024, ReLU/GELU,
      dropout 0.2/0.4, lr 1e-3/3e-4, LayerNorm sí/no. Cada configuración es un `B3_<...>.yaml`.
- [ ] Opcional: early stopping por mAP o F1 calibrado (cambia `entrenar.py`, que es de todos: avisen).
- [ ] Quitar el `xfail` de `test_idea3_gelu_y_layernorm`.

**Listo cuando** la mejor configuración está en el registro.

## Idea 4 · Focal Loss

FL(p_t) = −α_t (1 − p_t)^γ log(p_t): baja el peso de los aciertos fáciles (p = 0.9 → ×0.01 con γ = 2)
y el entrenamiento se va a los confusos (p = 0.3 → ×0.49).

- [ ] En `perdidas.py`: el tipo `{tipo: focal, gamma: 2}` (hoy lanza `NotImplementedError`).
      Decidan qué hacer con el `pos_weight` de la BCE (α por género) y documéntenlo.
- [ ] Experimentos `B4_focal_g1`, `_g2`, `_g3` (ya están en `experimentos/`). Los umbrales se
      recalibran solos al entrenar.
- [ ] Quitar el `xfail` de `test_idea4_focal_loss`.

**Listo cuando** γ = 1, 2 y 3 están evaluados en validación.

## Idea 5 · Pistas de texto (zero-shot)

CLIP no genera nada: solo compara el cartel con frases ("a poster of a mystery film"). Esos 17
números entran a la cabeza junto al vector de imagen.

- [ ] En `zeroshot.py`: `FRASES` (3 a 5 por género, en inglés), `similitudes` y `calcular`.
      `calcular` no vuelve a codificar carteles: usa `clip_b32__centro` (ya normalizados) × frases.
- [ ] `python -m clasificador.zeroshot clip_b32` → `vectores/clip_b32__zeroshot.npz` (N × 17).
- [ ] Variante `zeroshot` en `codificadores.py` (archivo de la Pareja A: cambio chico) para que el
      **predictor** calcule las pistas de un cartel nuevo. Debe dar lo mismo que `calcular`.
- [ ] Experimento `B5_zeroshot` (`vectores: [clip_b32__centro, clip_b32__zeroshot]`).
- [ ] Quitar el `xfail` de `test_idea5_zeroshot`.

**Listo cuando** el experimento imagen + pistas está en validación.

---

## Cuidados

- **Elijan siempre con validación, nunca con test** (`evaluar` se niega a medir test).
- Aunque no mejore, se registra y se sube: explicar por qué no funcionó también cuenta.
- **Para la demo:** si una cabeza nueva (GELU, LayerNorm) termina en un release, hay que llevarla
  a `construir_cabeza` de `p2m/clasificador.py` del repo principal
  (ver [Pareja C](PAREJA_C.md#integración-con-el-repo-principal)).
