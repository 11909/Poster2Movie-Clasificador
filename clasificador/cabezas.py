"""La parte entrenable del clasificador (Pareja B).

Mismo formato que `construir_cabeza` del repo principal, así un .pt de aquí
se puede evaluar allá y viceversa:

    {"tipo": "lineal"}                                   capa lineal entrada -> 17 (la base)
    {"tipo": "mlp", "ocultas": [512, 256], "dropout": 0.2}   MLP con ReLU y dropout

Claves opcionales del MLP (idea 3):
    "activacion": "relu" | "gelu"     (por defecto relu)
    "layernorm": true | false         (por defecto false; va entre la capa lineal y la activación)

Cada capa oculta es  Linear -> [LayerNorm] -> activación -> Dropout. Sin las claves nuevas la
estructura (y los nombres de los pesos en el .pt) es idéntica a la de antes: los .pt viejos cargan.

Una clave que no se conoce da error: si no, un .yaml con `activacion: gelu` entrenaría con
ReLU sin avisar. Para agregar un tipo o una clave, agréguenla aquí, documéntenla en docs/CONTRATOS.md,
y quiten el `xfail` de su prueba en tests/test_pendientes.py. Si el tipo nuevo no existe en el
repo principal, al integrar hay que llevar también este archivo.
"""

import torch

CLAVES = {"lineal": {"tipo"}, "mlp": {"tipo", "ocultas", "dropout", "activacion", "layernorm"}}
ACTIVACIONES = {"relu": torch.nn.ReLU, "gelu": torch.nn.GELU}


def construir_cabeza(arquitectura: dict | None, entrada: int, salida: int) -> torch.nn.Module:
    arquitectura = arquitectura or {"tipo": "lineal"}
    tipo = arquitectura["tipo"]
    if tipo not in CLAVES:
        raise ValueError(f"Arquitectura desconocida: {tipo}")
    sobran = set(arquitectura) - CLAVES[tipo]
    if sobran:
        raise ValueError(f"Claves que la cabeza '{tipo}' no conoce: {sorted(sobran)}")
    if tipo == "lineal":
        return torch.nn.Linear(entrada, salida)

    activacion = arquitectura.get("activacion", "relu")
    if activacion not in ACTIVACIONES:
        raise ValueError(f"Activación desconocida: {activacion!r} (opciones: {sorted(ACTIVACIONES)}).")
    layernorm = arquitectura.get("layernorm", False)
    if not isinstance(layernorm, bool):
        raise ValueError(f"layernorm debe ser true o false, no {layernorm!r}.")
    dropout = arquitectura.get("dropout", 0.2)
    if not 0 <= dropout < 1:
        raise ValueError(f"dropout debe estar en [0, 1), no {dropout!r}.")
    capas, anterior = [], entrada
    for oculta in arquitectura["ocultas"]:
        capas.append(torch.nn.Linear(anterior, oculta))
        if layernorm:
            capas.append(torch.nn.LayerNorm(oculta))
        capas += [ACTIVACIONES[activacion](), torch.nn.Dropout(dropout)]
        anterior = oculta
    return torch.nn.Sequential(*capas, torch.nn.Linear(anterior, salida))


def cargar_cabeza(guardado: dict, entrada: int) -> torch.nn.Module:
    cabeza = construir_cabeza(guardado.get("arquitectura"), entrada, len(guardado["generos"]))
    cabeza.load_state_dict(guardado["capa"])
    return cabeza.eval()
