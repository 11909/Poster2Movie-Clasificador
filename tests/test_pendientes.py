"""Lo que falta, idea por idea. Cada prueba describe lo mínimo que debe cumplir la idea.

Están marcadas `xfail(strict=True)`: hoy fallan "a propósito" (pytest las muestra como x).
Cuando una pareja implementa su idea y la prueba pasa, pytest la marca como error (XPASS):
quiten el `@pendiente(...)` de esa prueba en el mismo Pull Request. Así `pytest -q` siempre
dice qué falta. Pueden ajustar una prueba si su diseño es otro, explicándolo en el PR.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from clasificador import codificadores
from clasificador.cabezas import construir_cabeza
from clasificador.datos import GENEROS


def pendiente(quien):
    return pytest.mark.xfail(strict=True, reason=f"pendiente: {quien}")


# --- Lo que ya debe pasar: lo pendiente da errores claros, nunca corre en silencio

def test_lo_pendiente_avisa():
    with pytest.raises(NotImplementedError, match="Pareja A"):
        codificadores.crear_codificador("clip_l14__centro")
    with pytest.raises(NotImplementedError, match="idea 1"):
        codificadores.crear_codificador("clip_b32__3recortes")
    with pytest.raises(NotImplementedError, match="idea 5"):
        codificadores.crear_codificador("clip_b32__zeroshot")
    with pytest.raises(ValueError):
        construir_cabeza({"tipo": "mlp", "ocultas": [8], "activasion": "gelu"}, 4, 17)  # clave mal escrita


def test_muestra_fija():
    ids = range(1, 5001)
    assert codificadores.muestra_ids(ids, 100) == codificadores.muestra_ids(reversed(ids), 100)
    assert len(codificadores.muestra_ids(ids, 100)) == 100


# --- Pareja A

CARTEL = Image.new("RGB", (384, 576), (0, 0, 0))
CARTEL.paste((255, 0, 0), (0, 0, 384, 100))      # franja roja arriba (título)
CARTEL.paste((0, 0, 255), (0, 476, 384, 576))    # franja azul abajo


@pendiente("Pareja A, idea 1")
def test_idea1_relleno_y_tres_recortes():
    assert {"relleno", "3recortes"} <= set(codificadores.VARIANTES)
    [relleno] = codificadores.recortes(CARTEL, "relleno")
    assert relleno.width == relleno.height  # cuadrado, con el cartel completo
    arriba, centro, abajo = codificadores.recortes(CARTEL, "3recortes")
    assert all(r.width == r.height for r in (arriba, centro, abajo))
    assert arriba.getpixel((arriba.width // 2, 5)) == (255, 0, 0)          # se ve el título
    assert abajo.getpixel((abajo.width // 2, abajo.height - 5)) == (0, 0, 255)


@pendiente("Pareja A, idea 2")
def test_idea2_ocr_registrado():
    assert "ocr" in codificadores.VARIANTES and "ocr" not in codificadores.VARIANTES_PENDIENTES


@pendiente("Pareja A, idea 7a")
def test_idea7a_codificadores_nuevos():
    assert {"clip_l14", "siglip_base"} <= set(codificadores.CODIFICADORES)


# --- Pareja B

@pendiente("Pareja B, idea 3")
def test_idea3_gelu_y_layernorm():
    cabeza = construir_cabeza({"tipo": "mlp", "ocultas": [32, 16], "activacion": "gelu",
                               "layernorm": True, "dropout": 0.2}, 64, 17)
    tipos = {type(m) for m in cabeza.modules()}
    assert torch.nn.GELU in tipos and torch.nn.LayerNorm in tipos
    assert cabeza(torch.zeros(3, 64)).shape == (3, 17)


@pendiente("Pareja B, idea 4")
def test_idea4_focal_loss():
    from clasificador.perdidas import construir_perdida

    y = np.zeros((50, 17)); y[:, 0] = 1
    logits = torch.full((50, 17), -4.0); logits[:, 0] = 4.0   # aciertos fáciles
    objetivo = torch.tensor(y, dtype=torch.float32)
    focal = construir_perdida({"tipo": "focal", "gamma": 2}, y)(logits, objetivo)
    sin_foco = construir_perdida({"tipo": "focal", "gamma": 0}, y)(logits, objetivo)
    assert focal.ndim == 0 and torch.isfinite(focal)
    assert focal < sin_foco   # los fáciles casi no cuentan


@pendiente("Pareja B, idea 5")
def test_idea5_zeroshot():
    from clasificador import zeroshot

    assert set(zeroshot.FRASES) == set(GENEROS)
    assert all(3 <= len(f) <= 5 for f in zeroshot.FRASES.values())
    rng = np.random.default_rng(0)
    frases = {g: rng.normal(size=(3, 8)) for g in GENEROS}
    frases = {g: v / np.linalg.norm(v, axis=1, keepdims=True) for g, v in frases.items()}
    misterio = frases["Mystery"].mean(axis=0)
    imagen = (misterio / np.linalg.norm(misterio))[None]
    s = zeroshot.similitudes(imagen, frases)
    assert s.shape == (1, 17) and GENEROS[int(s.argmax())] == "Mystery"
    assert "zeroshot" in codificadores.VARIANTES


# --- Pareja C

@pendiente("Pareja C, idea 6")
def test_idea6_pliegues_por_franquicia():
    from clasificador.apilamiento import pliegues

    meta = pd.DataFrame({"id": range(200),
                         "coleccion": [f"saga{i % 30}" if i % 3 else "" for i in range(200)]})
    pliegue = pliegues(meta, k=5)
    assert set(pliegue) == set(range(5))
    sagas = meta.assign(p=pliegue.to_numpy())[meta["coleccion"] != ""].groupby("coleccion")["p"].nunique()
    assert (sagas == 1).all()   # una saga nunca queda en dos pliegues


@pendiente("Pareja C, idea 7b")
def test_idea7b_alinear_predicciones():
    from clasificador.ensamble import alinear

    p1 = np.arange(3 * 17, dtype=float).reshape(3, 17)
    ids, probs = alinear({"a": (np.array([1, 2, 3]), p1), "b": (np.array([3, 1, 2]), p1[[2, 0, 1]])})
    assert list(ids) == [1, 2, 3] or sorted(ids) == [1, 2, 3]
    orden = [list(ids).index(i) for i in (1, 2, 3)]
    assert np.array_equal(probs[0][orden], probs[1][orden])
