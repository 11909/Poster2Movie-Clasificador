"""Pareja B: MLP con GELU/LayerNorm (idea 3), Focal Loss (idea 4) y pistas zero-shot (idea 5).

Todo con datos falsos: no necesita GPU, ni Drive, ni descargar CLIP.
"""

import json

import numpy as np
import pytest
import torch

from clasificador import codificadores, zeroshot
from clasificador.cabezas import cargar_cabeza, construir_cabeza
from clasificador.datos import GENEROS
from clasificador.perdidas import FocalLoss, construir_perdida


# --- Idea 3 · cabezas

def test_mlp_sin_nuevas_claves_no_cambia():
    """Sin activacion/layernorm la estructura es la de antes (los .pt viejos siguen cargando)."""
    cabeza = construir_cabeza({"tipo": "mlp", "ocultas": [8, 4], "dropout": 0.1}, 6, 17)
    tipos = [type(m) for m in cabeza]
    assert tipos == [torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout] * 2 + [torch.nn.Linear]
    assert list(cabeza.state_dict()) == ["0.weight", "0.bias", "3.weight", "3.bias", "6.weight", "6.bias"]


@pytest.mark.parametrize("activacion", ["relu", "gelu"])
@pytest.mark.parametrize("layernorm", [False, True])
def test_mlp_variantes(activacion, layernorm):
    cabeza = construir_cabeza({"tipo": "mlp", "ocultas": [16, 8], "activacion": activacion,
                               "layernorm": layernorm}, 12, 17)
    modulos = {type(m) for m in cabeza.modules()}
    assert (torch.nn.GELU in modulos) == (activacion == "gelu")
    assert (torch.nn.ReLU in modulos) == (activacion == "relu")
    assert (torch.nn.LayerNorm in modulos) == layernorm
    assert cabeza(torch.zeros(5, 12)).shape == (5, 17)


def test_layernorm_va_antes_de_la_activacion():
    cabeza = construir_cabeza({"tipo": "mlp", "ocultas": [8], "activacion": "gelu", "layernorm": True}, 4, 17)
    assert [type(m) for m in cabeza] == [torch.nn.Linear, torch.nn.LayerNorm, torch.nn.GELU,
                                         torch.nn.Dropout, torch.nn.Linear]


def test_mlp_valores_invalidos():
    with pytest.raises(ValueError, match="Activación"):
        construir_cabeza({"tipo": "mlp", "ocultas": [8], "activacion": "swish"}, 4, 17)
    with pytest.raises(ValueError, match="layernorm"):
        construir_cabeza({"tipo": "mlp", "ocultas": [8], "layernorm": "si"}, 4, 17)


def test_cabeza_gelu_layernorm_se_guarda_y_se_carga():
    arq = {"tipo": "mlp", "ocultas": [8], "activacion": "gelu", "layernorm": True}
    cabeza = construir_cabeza(arq, 4, 17).eval()
    guardado = {"arquitectura": arq, "generos": GENEROS, "capa": cabeza.state_dict()}
    x = torch.randn(3, 4)
    assert torch.allclose(cabeza(x), cargar_cabeza(guardado, 4)(x))


# --- Idea 4 · Focal Loss

def _datos(n=64, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 17)) < 0.2).astype(np.float64)
    logits = torch.tensor(rng.normal(size=(n, 17)), dtype=torch.float32)
    return y, logits, torch.tensor(y, dtype=torch.float32)


def test_focal_gamma_cero_es_la_bce_de_la_base():
    y, logits, objetivo = _datos()
    bce = construir_perdida({"tipo": "bce"}, y)(logits, objetivo)
    focal = construir_perdida({"tipo": "focal", "gamma": 0}, y)(logits, objetivo)
    assert torch.allclose(bce, focal, atol=1e-6)


def test_focal_baja_el_peso_de_los_faciles_y_conserva_los_confusos():
    """p = 0.9 -> x0.01 con gamma 2; p = 0.3 -> x0.49 (los números de docs/PAREJA_B.md)."""
    sin_peso = FocalLoss(gamma=2)
    for p, factor in ((0.9, 0.01), (0.3, 0.49)):
        logit = torch.tensor([[np.log(p / (1 - p))]], dtype=torch.float32)
        uno = torch.ones(1, 1)
        esperado = -np.log(p) * factor
        assert sin_peso(logit, uno).item() == pytest.approx(esperado, rel=1e-4)


def test_focal_mas_gamma_menos_perdida():
    y, logits, objetivo = _datos()
    valores = [construir_perdida({"tipo": "focal", "gamma": g}, y)(logits, objetivo).item() for g in (0, 1, 2, 3)]
    assert valores == sorted(valores, reverse=True)


def test_focal_pos_weight_por_defecto_y_sin_el():
    y, logits, objetivo = _datos()
    con = construir_perdida({"tipo": "focal", "gamma": 2}, y)
    sin = construir_perdida({"tipo": "focal", "gamma": 2, "pos_weight": False}, y)
    assert con.pos_weight is not None and sin.pos_weight is None
    assert con(logits, objetivo) > sin(logits, objetivo)   # los positivos pesan más


def test_focal_gradiente_finito_incluso_con_logits_extremos_y_gamma_chico():
    y = np.zeros((4, 17)); y[:, 0] = 1
    logits = torch.full((4, 17), -30.0); logits[:, 0] = 30.0
    logits.requires_grad_()
    for gamma in (0.5, 2):
        perdida = construir_perdida({"tipo": "focal", "gamma": gamma}, y)(logits, torch.tensor(y, dtype=torch.float32))
        perdida.backward()
        assert torch.isfinite(perdida) and torch.isfinite(logits.grad).all()
        logits.grad = None


def test_perdidas_claves_desconocidas():
    y = np.zeros((4, 17))
    with pytest.raises(ValueError):
        construir_perdida({"tipo": "focal", "gama": 2}, y)   # clave mal escrita
    with pytest.raises(ValueError):
        construir_perdida({"tipo": "focal", "gamma": -1}, y)
    with pytest.raises(ValueError):
        construir_perdida({"tipo": "hinge"}, y)


# --- Idea 5 · zero-shot

def _frases_falsas(d=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    frases = {g: rng.normal(size=(k, d)) for g in GENEROS}
    return {g: (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32) for g, v in frases.items()}


def test_similitudes_es_el_promedio_del_coseno_por_genero():
    frases = _frases_falsas()
    imagen = np.random.default_rng(1).normal(size=(5, 8)).astype(np.float32)
    imagen /= np.linalg.norm(imagen, axis=1, keepdims=True)
    s = zeroshot.similitudes(imagen, frases)
    assert s.shape == (5, 17) and s.dtype == np.float32
    for j, g in enumerate(GENEROS):
        assert np.allclose(s[:, j], (imagen @ frases[g].T).mean(axis=1), atol=1e-6)


def test_frases_cumplen_el_contrato():
    assert set(zeroshot.FRASES) == set(GENEROS)
    for genero, frases in zeroshot.FRASES.items():
        assert 3 <= len(frases) <= 5 and len(set(frases)) == len(frases), genero
        assert all(f.isascii() and f.strip() for f in frases), genero   # CLIP: inglés


def test_variante_zeroshot_registrada():
    assert "zeroshot" in codificadores.VARIANTES and "zeroshot" not in codificadores.VARIANTES_PENDIENTES
    [cartel] = codificadores.recortes(object(), "zeroshot")   # mismo recorte que centro
    assert cartel is not None


class _Entrada(dict):
    def to(self, _):
        return self


class _ProcesadorFalso:
    """Imagen = vector de 32 números (ya viene así); texto = vector fijo por frase."""

    def __call__(self, images=None, text=None, **_):
        if images is not None:
            return _Entrada(x=torch.tensor(np.stack(images), dtype=torch.float32))
        generador = [np.random.default_rng(abs(hash(t)) % 2**32).normal(size=32) for t in text]
        return _Entrada(x=torch.tensor(np.stack(generador), dtype=torch.float32))


class _ModeloFalso:
    config = type("C", (), {"projection_dim": 32})()

    def get_image_features(self, x):
        return x

    get_text_features = get_image_features


def test_predictor_y_calcular_dan_lo_mismo(entorno, monkeypatch):
    """El archivo `__zeroshot.npz` (entrenamiento) y CodificadorZeroshot (predictor) coinciden."""
    monkeypatch.setattr(codificadores, "cargar_clip", lambda modelo_hf: (_ProcesadorFalso(), _ModeloFalso()))
    monkeypatch.setattr(codificadores, "DISPOSITIVO", "cpu")
    from clasificador import rutas
    from clasificador.datos import cargar_vectores

    zeroshot.calcular("clip_b32")
    ruta = rutas.carpeta_vectores() / "clip_b32__zeroshot.npz"
    ids, pistas = cargar_vectores("clip_b32__zeroshot")
    meta = json.loads(ruta.with_suffix(".json").read_text())
    assert pistas.shape == (600, 17) and meta["dim"] == 17 and meta["variante"] == "zeroshot"

    # los vectores `centro` falsos del conftest son la "salida de CLIP"; el codificador recibe
    # esos mismos números como si fueran imágenes y debe regresar las mismas pistas
    ids_centro, centro = cargar_vectores("clip_b32__centro")
    codificador = codificadores.crear_codificador("clip_b32__zeroshot")
    assert codificador.dim == 17
    por_codificador = codificador.codificar(list(centro[:20]))
    fila = {int(i): n for n, i in enumerate(ids)}
    esperado = pistas[[fila[int(i)] for i in ids_centro[:20]]]
    assert np.allclose(por_codificador, esperado, atol=1e-5)


def test_experimento_b5_entrena_con_imagen_y_pistas(entorno, monkeypatch, escribir_experimento):
    monkeypatch.setattr(codificadores, "cargar_clip", lambda modelo_hf: (_ProcesadorFalso(), _ModeloFalso()))
    monkeypatch.setattr(codificadores, "DISPOSITIVO", "cpu")
    from clasificador.entrenar import cargar_experimento, entrenar

    zeroshot.calcular("clip_b32")
    ruta = escribir_experimento(entorno, "B5_prueba", {"tipo": "lineal"},
                                vectores=("clip_b32__centro", "clip_b32__zeroshot"))
    modelo = torch.load(entrenar(cargar_experimento(ruta)), weights_only=False)
    assert modelo["capa"]["weight"].shape == (17, 32 + 17)
    assert modelo["vectores"] == ["clip_b32__centro", "clip_b32__zeroshot"]


def test_predictor_con_modelo_b5_usa_imagen_y_pistas(entorno, monkeypatch, escribir_experimento):
    """Lo que ve la demo: un .pt entrenado con [centro, zeroshot] predice un cartel nuevo y da las
    mismas probabilidades que la cabeza sobre los vectores guardados (entrenamiento = predictor)."""
    monkeypatch.setattr(codificadores, "cargar_clip", lambda modelo_hf: (_ProcesadorFalso(), _ModeloFalso()))
    monkeypatch.setattr(codificadores, "DISPOSITIVO", "cpu")
    from clasificador import rutas
    from clasificador.datos import cargar_vectores
    from clasificador.entrenar import cargar_experimento, entrenar
    from clasificador.predictor import Clasificador

    zeroshot.calcular("clip_b32")
    ruta = escribir_experimento(entorno, "B5_demo", {"tipo": "mlp", "ocultas": [16], "activacion": "gelu"},
                                vectores=("clip_b32__centro", "clip_b32__zeroshot"))
    modelo = entrenar(cargar_experimento(ruta))
    clasificador = Clasificador(modelo)
    ids, centro = cargar_vectores("clip_b32__centro")
    ids_z, pistas = cargar_vectores("clip_b32__zeroshot")
    assert list(ids) == list(ids_z)

    guardado = torch.load(modelo, weights_only=False)
    cabeza = cargar_cabeza(guardado, 32 + 17)
    unitario = zeroshot.normalizar(centro[:1])   # el predictor normaliza; los vectores reales ya lo están
    esperado = torch.sigmoid(cabeza(torch.tensor(np.concatenate([unitario, pistas[:1]], axis=1)))).detach().numpy()[0]
    resultado = clasificador.predecir(centro[0])   # el \"cartel\" falso ya es un vector de 32 números
    obtenido = np.array([resultado["probabilidades"][g] for g in GENEROS])
    assert np.allclose(obtenido, esperado, atol=1e-5)
    assert resultado["generos"], "toda película tiene al menos un género"


# --- Los tres juntos, entrenando de verdad (datos falsos)

@pytest.mark.parametrize("cabeza,perdida", [
    ({"tipo": "mlp", "ocultas": [32, 16], "activacion": "gelu", "layernorm": True, "dropout": 0.2}, {"tipo": "bce"}),
    ({"tipo": "lineal"}, {"tipo": "focal", "gamma": 2}),
    ({"tipo": "mlp", "ocultas": [32], "activacion": "gelu"}, {"tipo": "focal", "gamma": 1, "pos_weight": False}),
])
def test_entrenar_y_evaluar_con_las_ideas_de_b(entorno, cabeza, perdida):
    from clasificador.entrenar import cargar_experimento, entrenar
    from clasificador.evaluar import evaluar

    ruta = entorno / "B_prueba.yaml"
    ruta.write_text(f"nombre: B_prueba\npareja: B\nvectores: [clip_b32__centro]\ncabeza: {json.dumps(cabeza)}\n"
                    f"perdida: {json.dumps(perdida)}\nentrenamiento: {{epocas: 30, paciencia: 5}}\n")
    entrenar(cargar_experimento(ruta))
    resultado = evaluar("B_prueba", "val")
    assert resultado["F1_macro"] > resultado["trivial"]["F1_macro"]


def test_ranking_ordena_contra_la_base(entorno, escribir_experimento):
    from clasificador.entrenar import cargar_experimento, entrenar
    from clasificador.evaluar import evaluar
    from clasificador.ranking import ranking

    for nombre, cabeza in (("base", {"tipo": "lineal"}), ("B3_chico", {"tipo": "mlp", "ocultas": [8]}),
                           ("B3_grande", {"tipo": "mlp", "ocultas": [64], "activacion": "gelu"})):
        entrenar(cargar_experimento(escribir_experimento(entorno, nombre, cabeza)))
        evaluar(nombre, "val")
    tabla = ranking("B3_")
    assert set(tabla.index) == {"B3_chico", "B3_grande"}       # la base no entra, solo se compara
    assert tabla["cumple"].dtype == bool
    assert (tabla["cumple"] == ((tabla["d_macro"] >= 0) & (tabla["d_debiles"] > 0))).all()
    assert list(tabla["cumple"]) == sorted(tabla["cumple"], reverse=True)   # los que cumplen, primero
    with pytest.raises(FileNotFoundError, match="sin_base"):
        ranking("B3_", base="sin_base")


def test_diagnostico_de_pistas(entorno, monkeypatch):
    monkeypatch.setattr(codificadores, "cargar_clip", lambda modelo_hf: (_ProcesadorFalso(), _ModeloFalso()))
    monkeypatch.setattr(codificadores, "DISPOSITIVO", "cpu")
    zeroshot.calcular("clip_b32")
    tabla = zeroshot.diagnostico("clip_b32")
    assert sorted(tabla["genero"]) == sorted(GENEROS) and tabla["debil"].sum() == 4
    assert tabla["AUC"].dropna().between(0, 1).all()
