"""Apilamiento (stacking) — Pareja C, idea 6 — contrato 4.

Una etapa 2 corrige las 17 probabilidades de la etapa 1 con cómo se juntan los géneros
(Misterio casi siempre va con Thriller o Crimen). Se guarda como un combinado (contrato 4):
el predictor y `evaluar` ya la aplican.

    python -m clasificador.apilamiento C6_apilamiento base
    python -m clasificador.evaluar C6_apilamiento
    python -m clasificador.comparar base C6_apilamiento

Pasos:
1. `pliegues`: parte train en k pedazos POR FRANQUICIA (columna `coleccion` de la metadata;
   sin colección, la película sola), así una saga nunca queda en dos pedazos. Si no, la etapa 2
   aprendería de películas casi iguales a las que predice.
2. Predicciones FUERA DE PLIEGUE: por cada miembro y cada pliegue se entrena la etapa 1 con su
   misma receta (`entrenar.entrenar_cabeza`, su `.yaml`) en los otros k-1 pedazos y se predice el
   pedazo que no vio. Así la etapa 2 aprende con probabilidades tan "honestas" como las que verá
   en val, test y la demo. Con varios miembros se promedian con sus pesos (como en el ensamble).
3. Etapa 2: `cabezas.construir_cabeza(arquitectura, 17, 17)` (lineal por defecto; `--etapa2 mlp`).
   Con `residual: True` su salida se SUMA al logit de la probabilidad original y arranca en ceros:
   empieza siendo la etapa 1 tal cual y solo ajusta. Se regulariza con L2 hacia cero (= no
   corregir) y se detiene con la pérdida en val.
4. En val se usan los miembros entrenados con TODO train (sus `.pt`), igual que en la demo. Los
   umbrales se calibran en val sobre la salida final.
5. Guarda el combinado `<P2M_SALIDAS>/modelos/<nombre>.pt`, las predicciones fuera de pliegue en
   `<P2M_SALIDAS>/predicciones/<nombre>_oof.npz` y, si la etapa 2 es lineal, su matriz 17 × 17 en
   `resultados/<nombre>/pesos_etapa2.csv` (filas: género corregido; columnas: probabilidad que
   entra). El mapa de calor (`mapa_calor`) sale de esa matriz.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from clasificador import combinado, rutas
from clasificador.cabezas import construir_cabeza
from clasificador.datos import GENEROS, aplicar_umbrales, calibrar_umbrales, cargar_split
from clasificador.entrenar import commit_actual, entrenar_cabeza
from p2m import config, datos

ETAPA2_BASE = {"l2": 1e-3, "tasa": 1e-2, "epocas": 2000, "paciencia": 50, "semilla": 42}


# --------------------------------------------------------------------------- pliegues

def pliegues(meta_train: pd.DataFrame, k: int = 5, semilla: int = 42) -> pd.Series:
    """Número de pliegue (0..k-1) de cada película de train. Toda una franquicia (columna
    `coleccion`; sin colección, la película sola) cae en el MISMO pliegue.

    Reparto: las franquicias se barajan con la semilla y luego, de la más grande a la más chica,
    cada una va al pliegue que lleva menos películas. Así los pliegues quedan casi del mismo tamaño
    y el resultado no depende del orden de las filas. Regresa una Serie con el índice de `meta_train`.
    """
    if k < 2:
        raise ValueError("Se necesitan al menos 2 pliegues.")
    if "coleccion" in meta_train:
        coleccion = meta_train["coleccion"].fillna("").astype(str)
    else:
        print("AVISO: la metadata no trae la columna 'coleccion'; cada película va sola.")
        coleccion = pd.Series("", index=meta_train.index)
    grupo = pd.Series([config.grupo(int(i), c) for i, c in zip(meta_train["id"], coleccion)],
                      index=meta_train.index)

    tamanos = grupo.value_counts()
    if len(tamanos) < k:
        raise ValueError(f"Hay {len(tamanos)} franquicias/películas: no alcanzan para {k} pliegues.")
    nombres = np.array(sorted(tamanos.index))
    np.random.default_rng(semilla).shuffle(nombres)  # desempate al azar (fijo) entre iguales
    orden = sorted(nombres, key=lambda g: -tamanos[g])  # sorted es estable

    cuenta = np.zeros(k, dtype=np.int64)
    pliegue_de = {}
    for g in orden:
        p = int(np.argmin(cuenta))
        pliegue_de[g] = p
        cuenta[p] += tamanos[g]
    return grupo.map(pliegue_de).astype(int).rename("pliegue")


# --------------------------------------------------------------------------- etapa 1

def cargar_miembro(nombre: str) -> dict:
    ruta = rutas.ruta_modelo(nombre)
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro {ruta}: entrena antes el experimento '{nombre}'.")
    guardado = torch.load(ruta, map_location="cpu", weights_only=False)
    if combinado.es_combinado(guardado):
        raise ValueError(f"'{nombre}' ya es un combinado: apila sobre sus miembros directamente.")
    if list(guardado["generos"]) != list(GENEROS):
        raise ValueError(f"'{nombre}' tiene otros géneros u otro orden de géneros.")
    experimento = guardado.get("experimento") or {}
    if not {"cabeza", "perdida", "entrenamiento", "vectores"} <= experimento.keys():
        raise ValueError(f"'{nombre}' no guarda su receta (experimento): no se puede reentrenar por pliegues.")
    return guardado


def fuera_de_pliegue(experimento: dict, x: np.ndarray, y: np.ndarray, pliegue: np.ndarray,
                     x_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Probabilidades de train (n × 17) donde cada fila la predice un modelo que NO la vio."""
    oof = np.full((len(x), len(GENEROS)), np.nan, dtype=np.float32)
    for p in np.unique(pliegue):
        fuera = pliegue == p
        cabeza, mejor, epoca = entrenar_cabeza(experimento, x[~fuera], y[~fuera], x_val, y_val)
        with torch.no_grad():
            oof[fuera] = torch.sigmoid(cabeza(torch.tensor(x[fuera]))).numpy()
        print(f"  pliegue {p}: entrena {int((~fuera).sum())}, predice {int(fuera.sum())} "
              f"(época {epoca}, F1 macro val 0.5 = {mejor:.4f})")
    assert not np.isnan(oof).any()
    return oof


def predicciones_train(miembros: dict[str, dict], k: int, semilla: int):
    """(ids, y, pliegue, {miembro: oof}) de las películas de train que están en TODOS los miembros."""
    splits = {}
    for nombre, guardado in miembros.items():
        vectores = guardado["experimento"]["vectores"]
        splits[nombre] = (cargar_split("train", vectores), cargar_split("val", vectores))

    comunes = set.intersection(*(set(tr[2]) for tr, _ in splits.values()))
    ids = np.array(sorted(comunes), dtype=np.int64)
    meta = datos.metadata("train").set_index("id").loc[ids].reset_index()
    pliegue = pliegues(meta, k, semilla).to_numpy()
    print(f"Train en común: {len(ids)} películas en {k} pliegues de "
          f"{', '.join(str(int((pliegue == p).sum())) for p in range(k))}")

    oof, y = {}, None
    for nombre, ((x, y_m, ids_m), (x_val, y_val, _)) in splits.items():
        fila = {int(i): n for n, i in enumerate(ids_m)}
        orden = [fila[int(i)] for i in ids]
        x, y_m = x[orden], y_m[orden]
        if y is not None and not np.array_equal(y, y_m):
            raise ValueError("Los miembros no coinciden en las etiquetas de train.")
        y = y_m
        print(f"{nombre}: predicciones fuera de pliegue")
        oof[nombre] = fuera_de_pliegue(miembros[nombre]["experimento"], x, y, pliegue, x_val, y_val)
    return ids, y, pliegue, oof


# --------------------------------------------------------------------------- etapa 2

def _logit(p: torch.Tensor) -> torch.Tensor:
    return torch.logit(p.clamp(1e-6, 1 - 1e-6))


def construir_etapa2(arquitectura: dict) -> torch.nn.Module:
    """La etapa 2 en ceros en su última capa: con residual, al inicio no cambia nada."""
    etapa2 = construir_cabeza(arquitectura, len(GENEROS), len(GENEROS))
    ultima = etapa2 if isinstance(etapa2, torch.nn.Linear) else etapa2[-1]
    torch.nn.init.zeros_(ultima.weight)
    torch.nn.init.zeros_(ultima.bias)
    return etapa2


def entrenar_etapa2(p_train: np.ndarray, y_train: np.ndarray, p_val: np.ndarray, y_val: np.ndarray,
                    arquitectura: dict | None = None, ajustes: dict | None = None) -> torch.nn.Module:
    """Entrena la etapa 2 residual: logit final = logit(p) + etapa2(p).

    Lote completo con Adam; pérdida BCE + l2 · Σ pesos² (sin sesgos): sin evidencia, no corrige.
    Se queda con la época de menor pérdida BCE en val."""
    arquitectura = arquitectura or {"tipo": "lineal"}
    a = {**ETAPA2_BASE, **(ajustes or {})}
    torch.manual_seed(a["semilla"])
    etapa2 = construir_etapa2(arquitectura)
    optimizador = torch.optim.Adam(etapa2.parameters(), lr=a["tasa"])
    bce = torch.nn.BCEWithLogitsLoss()
    pesos = [w for n, w in etapa2.named_parameters() if n.endswith("weight") and w.ndim == 2]

    pt, yt = torch.tensor(p_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    pv, yv = torch.tensor(p_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    base_t, base_v = _logit(pt), _logit(pv)

    with torch.no_grad():
        mejor = bce(base_v, yv).item()  # sin corregir
    inicial, mejor_estado, sin_mejorar = mejor, {k: v.clone() for k, v in etapa2.state_dict().items()}, 0
    for _ in range(a["epocas"]):
        etapa2.train()
        optimizador.zero_grad()
        perdida = bce(base_t + etapa2(pt), yt) + a["l2"] * sum((w ** 2).sum() for w in pesos)
        perdida.backward()
        optimizador.step()
        etapa2.eval()
        with torch.no_grad():
            val = bce(base_v + etapa2(pv), yv).item()
        if val < mejor - 1e-6:
            mejor, sin_mejorar = val, 0
            mejor_estado = {k: v.clone() for k, v in etapa2.state_dict().items()}
        else:
            sin_mejorar += 1
            if sin_mejorar >= a["paciencia"]:
                break
    etapa2.load_state_dict(mejor_estado)
    print(f"Etapa 2 ({arquitectura['tipo']}): pérdida BCE en val {inicial:.4f} -> {mejor:.4f}")
    return etapa2.eval()


# --------------------------------------------------------------------------- todo junto

def apilar(nombre: str, miembros: list[str], k: int = 5, pesos: list[float] | None = None,
           arquitectura: dict | None = None, ajustes: dict | None = None) -> Path:
    """Predicciones fuera de pliegue de los miembros, entrena la etapa 2, calibra umbrales en
    val y guarda el combinado `<P2M_SALIDAS>/modelos/<nombre>.pt` (contrato 4)."""
    from clasificador.evaluar import probabilidades_split

    if len(set(miembros)) != len(miembros):
        raise ValueError("Hay miembros repetidos.")
    if nombre in miembros:
        raise ValueError("El modelo apilado no puede llamarse igual que uno de sus miembros.")
    if pesos is not None and len(pesos) != len(miembros):
        raise ValueError("Debe haber un peso por miembro.")
    arquitectura = arquitectura or {"tipo": "lineal"}
    a = {**ETAPA2_BASE, **(ajustes or {})}

    guardados = {m: cargar_miembro(m) for m in miembros}
    w = np.ones(len(miembros)) if pesos is None else np.asarray(pesos, dtype=float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Los pesos deben ser >= 0 y no todos cero.")
    w = w / w.sum()

    # 1-2. train: fuera de pliegue
    ids, y_train, pliegue, oof = predicciones_train(guardados, k, a["semilla"])
    p_train = sum(peso * oof[m] for m, peso in zip(miembros, w))

    # 4. val: los miembros con todo train, combinados exactamente como en la demo
    sin_etapa2 = combinado.armar(list(guardados.values()), np.full(len(GENEROS), 0.5), pesos=w.tolist())
    p_val, y_val = probabilidades_split(sin_etapa2, "val")

    # 3. etapa 2
    etapa2 = entrenar_etapa2(p_train, y_train, p_val, y_val, arquitectura, a)
    apilamiento = {"arquitectura": arquitectura, "capa": etapa2.state_dict(), "residual": True}
    con_etapa2 = combinado.armar(list(guardados.values()), np.full(len(GENEROS), 0.5), pesos=w.tolist(),
                                 apilamiento=apilamiento)
    p_final, _ = probabilidades_split(con_etapa2, "val")
    umbrales = calibrar_umbrales(y_val, p_final)

    def f1(p, u):
        return f1_score(y_val, aplicar_umbrales(p, u), average="macro", zero_division=0)

    print(f"Validación: {len(y_val)} películas")
    print(f"  {'sin etapa 2':<20} F1 macro = {f1(p_val, calibrar_umbrales(y_val, p_val)):.4f}")
    print(f"  {nombre:<20} F1 macro = {f1(p_final, umbrales):.4f}  (apilado)")

    # 5. guardar
    experimento = {
        "nombre": nombre, "pareja": "C",
        "descripcion": f"Idea 6: apilamiento ({arquitectura['tipo']}, residual) sobre {', '.join(miembros)}.",
        "miembros": list(miembros), "pliegues": k, "etapa2": {"arquitectura": arquitectura, **a},
    }
    ruta = combinado.guardar(
        combinado.armar(list(guardados.values()), umbrales, pesos=w.tolist(), apilamiento=apilamiento,
                        experimento=experimento, commit=commit_actual()),
        rutas.ruta_modelo(nombre),
    )
    oof_ruta = rutas.carpeta_predicciones() / f"{nombre}_oof.npz"
    oof_ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez(oof_ruta, ids_train=ids, pliegue=pliegue, prob_train=p_train.astype(np.float32),
             **{f"prob_train__{m}": oof[m] for m in miembros})
    if arquitectura["tipo"] == "lineal":
        guardar_pesos(nombre, etapa2.weight.detach().numpy())
    print(f"Guardado en {ruta}")
    return ruta


def guardar_pesos(nombre: str, pesos: np.ndarray) -> Path:
    """Matriz 17 × 17 de la etapa 2 lineal: filas = género corregido, columnas = probabilidad que entra."""
    tabla = pd.DataFrame(pesos, index=GENEROS, columns=GENEROS)
    ruta = rutas.RESULTADOS / nombre / "pesos_etapa2.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, float_format="%.5f")
    return ruta


def mapa_calor(nombre: str) -> Path:
    """`resultados/<nombre>/mapa_calor.png` a partir de `pesos_etapa2.csv` (figura del reporte)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta = rutas.RESULTADOS / nombre
    pesos = pd.read_csv(carpeta / "pesos_etapa2.csv", index_col=0)
    etiquetas = [config.GENEROS_ES[g] for g in pesos.index]
    limite = float(np.abs(pesos.to_numpy()).max()) or 1.0
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(pesos.to_numpy(), cmap="RdBu_r", vmin=-limite, vmax=limite)
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=90)
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_xlabel("probabilidad que entra")
    ax.set_ylabel("género corregido")
    ax.set_title(f"{nombre}: etapa 2 (rojo = empuja hacia arriba)")
    fig.colorbar(im)
    ruta = carpeta / "mapa_calor.png"
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return ruta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nombre", help="nombre del modelo apilado, p. ej. C6_apilamiento")
    parser.add_argument("miembros", nargs="+", help="experimento(s) de la etapa 1, p. ej. base")
    parser.add_argument("--pliegues", type=int, default=5)
    parser.add_argument("--pesos", nargs="+", type=float, default=None,
                        help="uno por miembro (se normalizan). Sin pesos: promedio simple")
    parser.add_argument("--etapa2", choices=["lineal", "mlp"], default="lineal")
    parser.add_argument("--ocultas", nargs="+", type=int, default=[64], help="solo con --etapa2 mlp")
    parser.add_argument("--l2", type=float, default=ETAPA2_BASE["l2"], help="regularización de la etapa 2")
    parser.add_argument("--semilla", type=int, default=ETAPA2_BASE["semilla"])
    args = parser.parse_args()
    arquitectura = ({"tipo": "lineal"} if args.etapa2 == "lineal"
                    else {"tipo": "mlp", "ocultas": args.ocultas, "activacion": "gelu", "dropout": 0.1})
    apilar(args.nombre, args.miembros, args.pliegues, args.pesos, arquitectura,
           {"l2": args.l2, "semilla": args.semilla})
    if args.etapa2 == "lineal":
        print(f"Mapa de calor: {mapa_calor(args.nombre)}")
