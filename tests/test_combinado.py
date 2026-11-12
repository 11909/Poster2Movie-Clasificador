"""Contrato 4: ensamble y apilamiento. Que evaluar (con vectores) y el predictor (con carteles)
combinen igual, y que un .pt normal siga dando exactamente lo mismo que antes."""

import numpy as np
import pytest
import torch

from clasificador import combinado, predictor, rutas
from clasificador.cabezas import construir_cabeza
from clasificador.datos import GENEROS, cargar_vectores


class CodificadorFalso:
    """En vez de imágenes recibe ids de película y regresa sus vectores del .npz."""

    creados: list[str] = []

    def __init__(self, nombre):
        ids, x = cargar_vectores(nombre)
        self.fila = {int(i): n for n, i in enumerate(ids)}
        self.x, self.dim = x, x.shape[1]
        CodificadorFalso.creados.append(nombre)

    def codificar(self, ids):
        return self.x[[self.fila[i] for i in ids]]


@pytest.fixture
def modelos(entorno, escribir_experimento, monkeypatch):
    """Dos experimentos entrenados (uno por archivo de vectores) y el predictor con codificador falso."""
    from clasificador.entrenar import cargar_experimento, entrenar

    CodificadorFalso.creados = []
    monkeypatch.setattr(predictor, "crear_codificador", CodificadorFalso)
    rutas_pt = [
        entrenar(cargar_experimento(escribir_experimento(entorno, "base", {"tipo": "lineal"}))),
        entrenar(cargar_experimento(escribir_experimento(
            entorno, "A_otro", {"tipo": "mlp", "ocultas": [16]}, ["otro__centro"]))),
    ]
    return [torch.load(r, weights_only=False) for r in rutas_pt]


def prob_de(guardado, ids, archivo):
    """Probabilidades de un .pt normal, calculadas a mano."""
    fake = CodificadorFalso(archivo)
    cabeza = construir_cabeza(guardado["arquitectura"], fake.dim, len(GENEROS))
    cabeza.load_state_dict(guardado["capa"])
    with torch.no_grad():
        return torch.sigmoid(cabeza.eval()(torch.tensor(fake.codificar(ids)))).numpy()


IDS = [3, 50, 499, 512, 600]


def test_pt_normal_da_lo_mismo_que_antes(modelos):
    base, _ = modelos
    clasificador = predictor.Clasificador(rutas.ruta_modelo("base"))
    assert np.array_equal(clasificador.probabilidades(IDS), prob_de(base, IDS, "clip_b32__centro"))
    assert np.array_equal(clasificador.umbrales, np.asarray(base["umbrales"]))


def test_ensamble_con_pesos(modelos):
    from clasificador.evaluar import evaluar
    from clasificador.release import preparar

    base, otro = modelos
    combinado.guardar(combinado.armar(modelos, np.full(17, 0.4), pesos=[3, 1],
                                      experimento={"nombre": "C_ensamble", "pareja": "C"}),
                      rutas.ruta_modelo("C_ensamble"))
    CodificadorFalso.creados = []
    clasificador = predictor.Clasificador(rutas.ruta_modelo("C_ensamble"))
    esperado = 0.75 * prob_de(base, IDS, "clip_b32__centro") + 0.25 * prob_de(otro, IDS, "otro__centro")
    assert np.allclose(clasificador.probabilidades(IDS), esperado, atol=1e-6)
    assert sorted(CodificadorFalso.creados[:2]) == ["clip_b32__centro", "otro__centro"]  # uno por archivo
    resultado = clasificador.predecir(IDS[0])  # misma interfaz
    assert set(resultado) == {"generos", "probabilidades"} and resultado["generos"]

    # se evalúa y se publica como cualquier experimento
    val = evaluar("C_ensamble", "val")
    assert val["pareja"] == "C" and val["vectores"] == ["clip_b32__centro", "otro__centro"]
    preparar("v0.1.0", "C_ensamble", "prueba")
    assert (rutas.RELEASES / "v0.1.0" / "C_ensamble.pt").exists()


def test_evaluar_y_predictor_combinan_igual(modelos):
    from clasificador.datos import cargar_split_por_archivo
    from clasificador.evaluar import probabilidades_split

    guardado = combinado.armar(modelos, np.full(17, 0.4), pesos=[1, 2])
    _, _, ids = cargar_split_por_archivo("val", guardado["vectores"])
    ruta = rutas.ruta_modelo("C_tmp")
    combinado.guardar(guardado, ruta)
    clasificador = predictor.Clasificador(ruta)
    prob_eval, _ = probabilidades_split(guardado, "val")
    assert np.allclose(prob_eval, clasificador.probabilidades(ids), atol=1e-6)


def test_apilamiento(modelos):
    base, _ = modelos
    etapa2 = construir_cabeza({"tipo": "lineal"}, 17, 17)
    torch.nn.init.zeros_(etapa2.weight)
    torch.nn.init.zeros_(etapa2.bias)
    # residual con etapa 2 en ceros = sin apilamiento (solo ajusta)
    neutro = combinado.armar([base], base["umbrales"],
                             apilamiento={"arquitectura": {"tipo": "lineal"}, "capa": etapa2.state_dict()})
    combinado.guardar(neutro, rutas.ruta_modelo("C_apila"))
    sin = predictor.Clasificador(rutas.ruta_modelo("base")).probabilidades(IDS)
    con = predictor.Clasificador(rutas.ruta_modelo("C_apila")).probabilidades(IDS)
    assert np.allclose(sin, con, atol=1e-5)

    # una etapa 2 que sí ajusta cambia las probabilidades
    torch.nn.init.constant_(etapa2.bias, 1.0)
    ajustado = combinado.armar([base], base["umbrales"],
                               apilamiento={"arquitectura": {"tipo": "lineal"}, "capa": etapa2.state_dict()})
    combinado.guardar(ajustado, rutas.ruta_modelo("C_apila2"))
    assert (predictor.Clasificador(rutas.ruta_modelo("C_apila2")).probabilidades(IDS) > sin).all()


def test_errores_claros(modelos):
    with pytest.raises(ValueError):
        combinado.armar(modelos, np.full(17, 0.4), pesos=[1])  # un peso por miembro
    with pytest.raises(ValueError):
        combinado.armar(modelos, np.full(5, 0.4))  # 17 umbrales
    with pytest.raises(ValueError):
        combinado.armar([], np.full(17, 0.4))
    hecho = combinado.armar(modelos, np.full(17, 0.4))
    with pytest.raises(ValueError):
        combinado.armar([hecho], np.full(17, 0.4))  # un combinado no es miembro de otro
