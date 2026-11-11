"""La parte entrenable del clasificador (Pareja B).

Mismo formato que `construir_cabeza` del repo principal, así un .pt de aquí
se puede evaluar allá y viceversa:

    {"tipo": "lineal"}                                   capa lineal entrada -> 17 (la base)
    {"tipo": "mlp", "ocultas": [512, 256], "dropout": 0.2}   MLP con ReLU y dropout

Para agregar un tipo (GELU, LayerNorm...), agréguenlo aquí y documenten sus
claves en el README. Si el tipo nuevo no existe en el repo principal, al
integrar hay que llevar también este archivo.
"""

import torch


def construir_cabeza(arquitectura: dict | None, entrada: int, salida: int) -> torch.nn.Module:
    arquitectura = arquitectura or {"tipo": "lineal"}
    if arquitectura["tipo"] == "lineal":
        return torch.nn.Linear(entrada, salida)
    if arquitectura["tipo"] == "mlp":
        capas, anterior = [], entrada
        for oculta in arquitectura["ocultas"]:
            capas += [torch.nn.Linear(anterior, oculta), torch.nn.ReLU(),
                      torch.nn.Dropout(arquitectura.get("dropout", 0.2))]
            anterior = oculta
        return torch.nn.Sequential(*capas, torch.nn.Linear(anterior, salida))
    raise ValueError(f"Arquitectura desconocida: {arquitectura['tipo']}")


def cargar_cabeza(guardado: dict, entrada: int) -> torch.nn.Module:
    cabeza = construir_cabeza(guardado.get("arquitectura"), entrada, len(guardado["generos"]))
    cabeza.load_state_dict(guardado["capa"])
    return cabeza.eval()
