"""Pistas de texto zero-shot — PENDIENTE (Pareja B, idea 5).

CLIP no genera nada: solo compara. Para cada cartel se mide qué tanto se parece a frases
como "a poster of a mystery film", y esos 17 números entran a la cabeza junto al vector.

Qué va aquí:
1. `FRASES`: de 3 a 5 frases por género (en inglés: CLIP se entrenó en inglés).
2. `similitudes`: similitud coseno cartel-frase, promediada por género -> n × 17.
3. `calcular`: archivo del contrato 1 `<codificador>__zeroshot.npz` (ids, vectores n × 17).
   No hace falta volver a codificar los carteles: los vectores `<codificador>__centro` ya
   están normalizados, así que basta con codificar las frases (una vez) y multiplicar.
4. En `codificadores.py`, la variante `zeroshot` (pendiente) para que el PREDICTOR también
   calcule las 17 similitudes de un cartel nuevo. Debe dar lo mismo que `calcular`.

Experimento:
    vectores: [clip_b32__centro, clip_b32__zeroshot]

Uso:
    python -m clasificador.zeroshot clip_b32
"""

import argparse

import numpy as np

from clasificador.datos import GENEROS

# PENDIENTE: de 3 a 5 frases por género, p. ej. "Mystery": ["a poster of a mystery film", ...]
FRASES: dict[str, list[str]] = {}


def similitudes(vectores_imagen: np.ndarray, vectores_frases: dict[str, np.ndarray]) -> np.ndarray:
    """(n, d) vectores de cartel normalizados + {género: (k, d) vectores de frases normalizados}
    -> (n, 17) similitud coseno promedio por género, en el orden de GENEROS."""
    raise NotImplementedError("Zero-shot: pendiente de la Pareja B (idea 5).")


def calcular(codificador: str = "clip_b32") -> None:
    """Escribe `<P2M_SALIDAS>/vectores/<codificador>__zeroshot.npz` (+ .json) a partir de
    `<codificador>__centro` y las frases codificadas con `codificar_texto`."""
    raise NotImplementedError("Zero-shot: pendiente de la Pareja B (idea 5).")


assert not FRASES or set(FRASES) == set(GENEROS), "FRASES debe tener los 17 géneros"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("codificador", nargs="?", default="clip_b32")
    calcular(parser.parse_args().codificador)
