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

`{"tipo": "mlp", "ocultas": [...], "dropout": ..., "activacion": "relu"|"gelu", "layernorm": true|false}`.
Cada capa oculta es `Linear → [LayerNorm] → activación → Dropout`. Una clave desconocida o un valor
inválido da error (no se ignora). Sin las claves nuevas la estructura es la de siempre: los `.pt`
viejos siguen cargando.

- [x] En `cabezas.py`: `activacion: gelu` y `layernorm: true`.
- [x] Rejilla desde el notebook: 1 a 3 capas ocultas, ancho 256/512/1024, ReLU/GELU,
      dropout 0.2/0.4, lr 1e-3/3e-4, LayerNorm sí/no (96 configuraciones; `RAPIDA = True` tamiza 24
      y afina las mejores). Cada configuración es un `B3_<...>.yaml`.
- [x] **Correr la rejilla** (`notebooks/B3_cabezas.ipynb`) y repetir la mejor con 3 semillas.
- [ ] Opcional: early stopping por mAP o F1 calibrado (cambia `entrenar.py`, que es de todos: avisen).
- [x] Quitar el `xfail` de `test_idea3_gelu_y_layernorm`.

**Listo cuando** la mejor configuración está en el registro (`python -m clasificador.ranking B3_`).

## Idea 4 · Focal Loss

FL(p_t) = −α_t (1 − p_t)^γ log(p_t): baja el peso de los aciertos fáciles (p = 0.9 → ×0.01 con γ = 2)
y el entrenamiento se va a los confusos (p = 0.3 → ×0.49).

- [x] En `perdidas.py`: el tipo `{tipo: focal, gamma: 2}` (`FocalLoss`, estable con logits extremos;
      γ = 0 es exactamente la BCE de la base).
- [x] **Decisión sobre el `pos_weight`:** se conserva por defecto como α por género. γ corrige qué tan
      fácil es el ejemplo, no qué tan raro es el género (Western 1 %); y así la única diferencia contra la
      base es γ. `pos_weight: false` lo quita y se mide en `B4_focal_g2_sinpw`. Razones completas en
      el docstring de `perdidas.py`.
- [x] **Correr** `B4_focal_g1`, `_g2`, `_g3` y `_g2_sinpw` (ya están en `experimentos/`). Los umbrales se
      recalibran solos al entrenar. Miren también el mAP: la focal comprime las probabilidades.
- [x] Quitar el `xfail` de `test_idea4_focal_loss`.

**Listo cuando** γ = 1, 2 y 3 están evaluados en validación.

## Idea 5 · Pistas de texto (zero-shot)

CLIP no genera nada: solo compara el cartel con frases ("a poster of a mystery film"). Esos 17
números entran a la cabeza junto al vector de imagen.

- [x] En `zeroshot.py`: `FRASES` (3 a 5 por género, en inglés; 5 en los cuatro débiles), `similitudes` y
      `calcular`. `calcular` no vuelve a codificar carteles: usa `clip_b32__centro` (ya normalizados) × frases.
- [x] **Correr** `python -m clasificador.zeroshot clip_b32` → `vectores/clip_b32__zeroshot.npz` (N × 17).
- [x] `python -m clasificador.zeroshot clip_b32 --diagnostico`: AUC de cada pista en validación. Si un
      género débil sale cerca de 0.5, reescriban sus frases (se afina con validación, nunca test).
- [x] Variante `zeroshot` en `codificadores.py` (`CodificadorZeroshot`, archivo de la Pareja A: cambio
      chico) para que el **predictor** calcule las pistas de un cartel nuevo. Una prueba verifica que da lo
      mismo que `calcular` y que un `.pt` imagen + pistas predice igual que en el entrenamiento.
- [x] **Correr** `B5_zeroshot` (cabeza lineal) y `B5_zeroshot_mlp` (`vectores: [clip_b32__centro, clip_b32__zeroshot]`).
- [x] Quitar el `xfail` de `test_idea5_zeroshot`.

**Qué esperar:** cada pista es una función *lineal* del vector de imagen, así que con una cabeza lineal
no agrega información, solo cambia el punto de partida y la regularización. Si `B5_zeroshot` queda igual
que la base no es un error: es lo que predice la teoría, y el MLP (`B5_zeroshot_mlp`) es donde pueden
mezclarse con la imagen. Registren y expliquen el resultado, sea cual sea.

**Listo cuando** el experimento imagen + pistas está en validación.

---

## Cuidados

- **Elijan siempre con validación, nunca con test** (`evaluar` se niega a medir test).
- **`python -m clasificador.ranking B3_`** (o `B4_`, `B5_`) ordena los experimentos contra la base con la regla
  de publicación: F1 macro no baja y el promedio de los 4 débiles sube (columna `cumple`).
- Con muchas configuraciones, la mejor en validación puede serlo por suerte: repitan la ganadora con otras
  semillas antes de decir que mejora (el notebook B3 lo hace).
- **Al final, junten lo que sí funcionó** (mejor MLP + focal si ayudó + pistas si ayudaron) en un experimento
  `B_combinado.yaml`: las tres ideas no se excluyen, y es el candidato natural a release.
- Aunque no mejore, se registra y se sube: explicar por qué no funcionó también cuenta.
- **Para la demo:** si una cabeza nueva (GELU, LayerNorm) termina en un release, hay que llevarla
  a `construir_cabeza` de `p2m/clasificador.py` del repo principal
  (ver [Pareja C](PAREJA_C.md#integración-con-el-repo-principal)).
