"""La parte entrenable del clasificador (Pareja B).

Mismo formato que `construir_cabeza` del repo principal, así un .pt de aquí
se puede evaluar allá y viceversa:

    {"tipo": "lineal"}                                   capa lineal entrada -> 17 (la base)
    {"tipo": "mlp", "ocultas": [512, 256], "dropout": 0.2}   MLP con ReLU y dropout

PENDIENTE (Pareja B, idea 3), claves opcionales del MLP:
    "activacion": "relu" | "gelu"     (hoy solo relu)
    "layernorm": true | false         (hoy solo false)

Una clave que no se conoce da error: si no, un .yaml con `activacion: gelu` entrenaría con
ReLU sin avisar. Para agregar un tipo o una clave, agréguenla aquí, documéntenla en el README,
y quiten el `xfail` de su prueba en tests/test_pendientes.py. Si el tipo nuevo no existe en el
repo principal, al integrar hay que llevar también este archivo.
"""

import torch

CLAVES = {"lineal": {"tipo"}, "mlp": {"tipo", "ocultas", "dropout", "activacion", "layernorm"}}


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

    if arquitectura.get("activacion", "relu") != "relu":
        raise NotImplementedError("activacion distinta de relu: pendiente de la Pareja B (idea 3).")
    if arquitectura.get("layernorm", False):
        raise NotImplementedError("layernorm: pendiente de la Pareja B (idea 3).")
    capas, anterior = [], entrada
    for oculta in arquitectura["ocultas"]:
        capas += [torch.nn.Linear(anterior, oculta), torch.nn.ReLU(),
                  torch.nn.Dropout(arquitectura.get("dropout", 0.2))]
        anterior = oculta
    return torch.nn.Sequential(*capas, torch.nn.Linear(anterior, salida))


def cargar_cabeza(guardado: dict, entrada: int) -> torch.nn.Module:
    cabeza = construir_cabeza(guardado.get("arquitectura"), entrada, len(guardado["generos"]))
    cabeza.load_state_dict(guardado["capa"])
    return cabeza.eval()
