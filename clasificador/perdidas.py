"""Funciones de pérdida (Pareja B).

    {"tipo": "bce"}                            BCE con pos_weight = negativos/positivos por género (la base)
    {"tipo": "focal", "gamma": 2}              Focal Loss (idea 4)
    {"tipo": "focal", "gamma": 2, "pos_weight": false}   Focal Loss sin el peso de positivos

Focal Loss (Lin et al., 2017), por cada par (película, género) con p = sigmoide(logit):

    FL = -[ α · y · (1 - p)^γ · log p  +  (1 - y) · p^γ · log(1 - p) ]

El factor (1 - p)^γ baja el peso de los aciertos fáciles (p = 0.9 → ×0.01 con γ = 2) y deja casi
intacto el de los confusos (p = 0.3 → ×0.49): el entrenamiento se va a donde el modelo duda.
Con γ = 0 es exactamente la BCE de la base.

Decisión sobre el pos_weight (α de la focal):
- Por defecto SE CONSERVA el `pos_weight` de la base como α por género (negativos/positivos).
  Razón: γ solo corrige "qué tan fácil es el ejemplo", no "qué tan raro es el género". Sin α, un
  Western (1 %) sigue teniendo ~99 % de ejemplos negativos fáciles y el modelo se inclina a decir
  «no». Además así la única diferencia contra la base es γ, y la comparación es limpia.
- `pos_weight: false` (α = 1) existe para medir lo contrario: ¿γ solo basta contra el desbalance?
  Ver `experimentos/B4_focal_g2_sinpw.yaml`.
- Efecto a vigilar: la focal con α grande empuja las probabilidades hacia arriba y las comprime.
  No importa para el F1 porque `entrenar` recalibra un umbral por género, pero sí para el mAP
  y para el ensamble (que promedia probabilidades): comparen mAP además de F1.
"""

import numpy as np
import torch
import torch.nn.functional as F

CLAVES = {"bce": {"tipo"}, "focal": {"tipo", "gamma", "pos_weight"}}
GAMMA_POR_DEFECTO = 2.0


class FocalLoss(torch.nn.Module):
    """Focal Loss binaria multi-etiqueta sobre logits, con promedio sobre todos los pares.

    `pos_weight`: tensor (17,) con α por género para los positivos, o None (α = 1).
    Se calcula con logsigmoid y no con (1 - p).clamp().log(): es estable con logits extremos y su
    gradiente es finito aun con γ < 1 (donde (1 - p)^γ no es derivable en p = 1).
    """

    def __init__(self, gamma: float = GAMMA_POR_DEFECTO, pos_weight: torch.Tensor | None = None):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma debe ser >= 0, no {gamma}.")
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight)  # None es válido: no entra al state_dict

    def forward(self, logits: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
        log_p = F.logsigmoid(logits)        # log p
        log_no_p = F.logsigmoid(-logits)    # log (1 - p)
        # (1 - p)^γ = exp(γ · log(1 - p)); p^γ = exp(γ · log p)
        positivo = -objetivo * torch.exp(self.gamma * log_no_p) * log_p
        if self.pos_weight is not None:
            positivo = positivo * self.pos_weight
        negativo = -(1 - objetivo) * torch.exp(self.gamma * log_p) * log_no_p
        return (positivo + negativo).mean()

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, pos_weight={'sí' if self.pos_weight is not None else 'no'}"


def construir_perdida(config: dict | None, y_train: np.ndarray) -> torch.nn.Module:
    config = config or {"tipo": "bce"}
    tipo = config["tipo"]
    if tipo not in CLAVES:
        raise ValueError(f"Pérdida desconocida: {tipo}")
    sobran = set(config) - CLAVES[tipo]
    if sobran:
        raise ValueError(f"Claves que la pérdida '{tipo}' no conoce: {sorted(sobran)}")

    positivos = y_train.sum(axis=0)
    # Sin este peso el modelo aprende a decir "no" a Western siempre y acierta el 99 %.
    pos_weight = torch.tensor((len(y_train) - positivos) / np.maximum(positivos, 1), dtype=torch.float32)
    if tipo == "bce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    gamma = config.get("gamma", GAMMA_POR_DEFECTO)
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or gamma < 0:
        raise ValueError(f"gamma debe ser un número >= 0, no {gamma!r}.")
    con_peso = config.get("pos_weight", True)
    if not isinstance(con_peso, bool):
        raise ValueError(f"pos_weight de la focal debe ser true o false, no {con_peso!r}.")
    return FocalLoss(gamma, pos_weight if con_peso else None)
