"""Funciones de pérdida (Pareja B).

    {"tipo": "bce"}                 BCE con pos_weight = negativos/positivos por género (la base)
    {"tipo": "focal", "gamma": 2}   Focal Loss — PENDIENTE (Pareja B, idea 4)
"""

import numpy as np
import torch


def construir_perdida(config: dict | None, y_train: np.ndarray) -> torch.nn.Module:
    config = config or {"tipo": "bce"}
    positivos = y_train.sum(axis=0)
    # Sin este peso el modelo aprende a decir "no" a Western siempre y acierta el 99 %.
    pos_weight = torch.tensor((len(y_train) - positivos) / np.maximum(positivos, 1), dtype=torch.float32)
    if config["tipo"] == "bce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if config["tipo"] == "focal":
        raise NotImplementedError("Focal Loss: pendiente de la Pareja B (idea 4).")
    raise ValueError(f"Pérdida desconocida: {config['tipo']}")
