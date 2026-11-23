"""Pistas de texto zero-shot (Pareja B, idea 5).

CLIP no genera nada: solo compara. Para cada cartel se mide qué tanto se parece a frases como
"a poster of a mystery film", y esos 17 números (uno por género) entran a la cabeza junto al
vector de imagen.

Qué hay aquí:
1. `FRASES`: de 3 a 5 frases por género (en inglés: CLIP se entrenó en inglés).
2. `similitudes`: similitud coseno cartel-frase, promediada por género -> n × 17.
3. `calcular`: escribe el archivo del contrato 1 `<codificador>__zeroshot.npz` (ids, vectores
   n × 17). No vuelve a codificar carteles: los vectores `<codificador>__centro` ya están
   normalizados, así que basta con codificar las frases (una vez) y multiplicar.
4. En `codificadores.py`, la variante `zeroshot` (`CodificadorZeroshot`) usa las mismas funciones
   (`codificar_frases` y `similitudes`) para que el PREDICTOR calcule las 17 similitudes de un
   cartel nuevo y dé lo mismo que `calcular`.

Experimento:
    vectores: [clip_b32__centro, clip_b32__zeroshot]

Uso:
    python -m clasificador.zeroshot clip_b32                 # calcula las pistas
    python -m clasificador.zeroshot clip_b32 --diagnostico   # ¿qué tan informativa es cada pista?

Qué esperar (para interpretar el resultado, funcione o no):
Cada pista es el promedio de productos punto entre el vector del cartel y unos vectores de texto
FIJOS, o sea una función LINEAL del vector de imagen. Una cabeza lineal sobre [imagen, pistas]
no puede representar nada que no pudiera ya sobre [imagen]: lo único que cambia es el punto de
partida y la regularización del entrenamiento (las pistas son una dirección «con significado»
que el modelo no tiene que descubrir con 60 mil ejemplos). Por eso la ganancia esperada es
modesta y se concentra en los géneros raros; con un MLP, que sí mezcla entradas, puede haber
más. Si no mejora, es un resultado válido: se registra y se explica.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from clasificador.datos import GENEROS

# 3 a 5 frases por género, en inglés. Mezclan la forma «a poster of a ... film» con pistas de lo
# que se VE en un cartel de ese género (el modelo solo mira la imagen). En los cuatro géneros
# débiles hay cinco, con las pistas visuales que los separan de Thriller, Drama/Guerra y Aventura.
FRASES: dict[str, list[str]] = {
    "Action": [
        "a poster of an action film",
        "a movie poster with explosions and a hero holding a gun",
        "an action movie poster with fighting and fast cars",
        "a poster of an action blockbuster",
    ],
    "Adventure": [
        "a poster of an adventure film",
        "a movie poster of explorers on a dangerous journey",
        "an adventure movie poster with treasure and exotic landscapes",
        "a poster of an epic adventure",
    ],
    "Animation": [
        "a poster of an animated film",
        "a cartoon movie poster with colorful characters",
        "a poster of a computer animated movie",
        "an animated movie poster for children",
    ],
    "Comedy": [
        "a poster of a comedy film",
        "a funny movie poster with a smiling cast",
        "a poster of a romantic comedy or a silly comedy",
        "a lighthearted comedy movie poster with bright colors",
    ],
    "Crime": [
        "a poster of a crime film",
        "a movie poster about gangsters and the mafia",
        "a crime movie poster with a criminal, a gun and a city at night",
        "a poster of a heist or a police crime story",
        "a gritty crime drama movie poster with a mugshot or a getaway car",
    ],
    "Documentary": [
        "a poster of a documentary film",
        "a documentary movie poster with real people and a title",
        "a poster of a non-fiction documentary about real events",
        "a documentary poster with archival photographs",
    ],
    "Drama": [
        "a poster of a drama film",
        "a serious dramatic movie poster with an emotional face",
        "a poster of an award-winning drama about human relationships",
        "a quiet dramatic movie poster with muted colors",
    ],
    "Family": [
        "a poster of a family film",
        "a family movie poster with kids and parents together",
        "a heartwarming family movie poster with a pet or a child",
        "a family friendly movie poster for all ages",
    ],
    "Fantasy": [
        "a poster of a fantasy film",
        "a fantasy movie poster with magic, dragons and wizards",
        "a poster of a magical world with castles and mythical creatures",
        "a fantasy movie poster with a glowing sword or a spell",
        "a poster of an enchanted fairy tale kingdom",
    ],
    "History": [
        "a poster of a historical film",
        "a period drama movie poster with people in old fashioned costumes",
        "a poster of a movie based on a true story from history",
        "a historical movie poster about kings, empires or famous leaders",
        "a vintage looking movie poster set in the past",
    ],
    "Horror": [
        "a poster of a horror film",
        "a scary movie poster with a monster in the dark",
        "a horror movie poster with blood and a terrified face",
        "a poster of a haunted house or a slasher movie",
    ],
    "Mystery": [
        "a poster of a mystery film",
        "a movie poster of a detective investigating a mystery",
        "a mystery movie poster with a shadowy figure and unanswered questions",
        "a poster of a whodunit with clues, a magnifying glass or a foggy street",
        "a dark mysterious movie poster with a hidden secret",
    ],
    "Romance": [
        "a poster of a romantic film",
        "a romance movie poster with a couple in love",
        "a love story movie poster with two people about to kiss",
        "a romantic movie poster with soft warm colors",
    ],
    "Science Fiction": [
        "a poster of a science fiction film",
        "a sci-fi movie poster with spaceships and planets",
        "a futuristic movie poster with robots and aliens",
        "a science fiction movie poster with a dystopian city",
    ],
    "Thriller": [
        "a poster of a thriller film",
        "a suspenseful thriller movie poster with a person running",
        "a tense psychological thriller movie poster",
        "a poster of a thriller with a dark city and a threatening figure",
    ],
    "War": [
        "a poster of a war film",
        "a war movie poster with soldiers on a battlefield",
        "a poster of a World War II movie with tanks and planes",
        "a war movie poster with a helmet and a bleak sky",
    ],
    "Western": [
        "a poster of a western film",
        "a western movie poster with cowboys and a desert town",
        "a poster of a gunslinger on horseback at sunset",
        "a wild west movie poster with a sheriff and a saloon",
    ],
}


def todas_las_frases() -> list[str]:
    """Las frases en un orden fijo (el de GENEROS), para codificarlas en un solo lote."""
    return [f for genero in GENEROS for f in FRASES[genero]]


def codificar_frases(codificador) -> dict[str, np.ndarray]:
    """{género: (k, d)} vectores de texto normalizados. `codificador` es un CodificadorCLIP
    (cualquiera con `codificar_texto`). Las frases se codifican una vez, en un solo lote."""
    vectores = np.asarray(codificador.codificar_texto(todas_las_frases()), dtype=np.float32)
    por_genero, inicio = {}, 0
    for genero in GENEROS:
        k = len(FRASES[genero])
        por_genero[genero] = vectores[inicio:inicio + k]
        inicio += k
    return por_genero


def similitudes(vectores_imagen: np.ndarray, vectores_frases: dict[str, np.ndarray]) -> np.ndarray:
    """(n, d) vectores de cartel normalizados + {género: (k, d) vectores de frases normalizados}
    -> (n, 17) similitud coseno promedio por género, en el orden de GENEROS (float32)."""
    imagen = np.asarray(vectores_imagen, dtype=np.float32)
    columnas = [(imagen @ np.asarray(vectores_frases[g], dtype=np.float32).T).mean(axis=1) for g in GENEROS]
    return np.stack(columnas, axis=1).astype(np.float32)


def normalizar(x: np.ndarray) -> np.ndarray:
    """Norma 1 por fila. Los vectores `centro` ya vienen así (no cambia nada); se hace aquí para
    que `calcular` y `CodificadorZeroshot.codificar` den lo mismo aunque no fuera cierto."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def calcular(codificador: str = "clip_b32") -> None:
    """Escribe `<P2M_SALIDAS>/vectores/<codificador>__zeroshot.npz` (+ .json) a partir de
    `<codificador>__centro` y las frases codificadas con `codificar_texto`."""
    from clasificador import rutas
    from clasificador.codificadores import crear_codificador
    from clasificador.datos import cargar_vectores

    modelo = crear_codificador(f"{codificador}__centro")
    if not hasattr(modelo, "codificar_texto"):
        raise ValueError(f"'{codificador}' no codifica texto: el zero-shot necesita CLIP o SigLIP.")
    ids, centro = cargar_vectores(f"{codificador}__centro")
    pistas = similitudes(normalizar(centro), codificar_frases(modelo))

    nombre = f"{codificador}__zeroshot"
    destino = rutas.carpeta_vectores() / f"{nombre}.npz"
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destino, ids=ids, vectores=pistas)
    destino.with_suffix(".json").write_text(json.dumps({
        "nombre": nombre, "codificador": codificador, "variante": "zeroshot",
        "modelo_hf": modelo.modelo_hf, "dim": len(GENEROS), "peliculas": int(len(ids)), "muestra": None,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frases": FRASES,
    }, indent=2, ensure_ascii=False))
    print(f"Guardadas {pistas.shape[1]} pistas de {len(ids)} películas en {destino}")


def diagnostico(codificador: str = "clip_b32") -> "pd.DataFrame":
    """AUC (en VALIDACIÓN) de la pista de cada género contra su etiqueta, sin entrenar nada.

    0.5 = la pista no distingue nada; 1.0 = separa perfecto. Sirve para decidir qué frases
    reescribir antes de gastar un experimento: se afinan FRASES mirando validación (nunca test).
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from clasificador.datos import GENEROS_DEBILES, cargar_split_por_archivo

    xs, y, _ = cargar_split_por_archivo("val", [f"{codificador}__zeroshot"])
    pistas = xs[f"{codificador}__zeroshot"]
    filas = []
    for j, genero in enumerate(GENEROS):
        positivos = int(y[:, j].sum())
        hay_ambos = 0 < positivos < len(y)
        filas.append({"genero": genero, "peliculas": positivos, "debil": genero in GENEROS_DEBILES,
                      "AUC": float(roc_auc_score(y[:, j], pistas[:, j])) if hay_ambos else float("nan")})
    return pd.DataFrame(filas).sort_values("AUC", ascending=False, ignore_index=True)


assert set(FRASES) == set(GENEROS), "FRASES debe tener los 17 géneros"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("codificador", nargs="?", default="clip_b32")
    parser.add_argument("--diagnostico", action="store_true", help="AUC por género de las pistas ya calculadas (val)")
    args = parser.parse_args()
    if args.diagnostico:
        print(diagnostico(args.codificador).to_string(index=False, float_format="{:.3f}".format))
    else:
        calcular(args.codificador)
