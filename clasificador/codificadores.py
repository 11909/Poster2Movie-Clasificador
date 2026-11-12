"""Codificadores: cartel -> vector (Pareja A). Produce los archivos del contrato 1.

Cada archivo de vectores se llama `<codificador>__<variante>`, por ejemplo
`clip_b32__centro`. La MISMA clase que genera el archivo es la que usa el
predictor, así el entrenamiento y la demo procesan igual el cartel.

Hoy existe:
    clip_b32 + centro   CLIP ViT-B/32, recorte central de 224 px (el de la base)

Pendiente (Pareja A):
    variantes `relleno` y `3recortes` (idea 1), `ocr` (idea 2)
    codificadores `clip_l14`, `siglip_base`, ... (idea 7a)

Uso:
    python -m clasificador.codificadores clip_b32 --variante centro
    python -m clasificador.codificadores clip_b32 --variante ocr --muestra 2000   # prueba chica

Para agregar algo pendiente: muévanlo de *_PENDIENTES a CODIFICADORES / VARIANTES, implementen
su clase o su recorte, y quiten el `xfail` de su prueba en tests/test_pendientes.py.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
import torch

from clasificador import rutas
from p2m import datos

# id corto -> id en Hugging Face
CODIFICADORES = {
    "clip_b32": "openai/clip-vit-base-patch32",
}
VARIANTES = ["centro"]

# Lo que falta, con quién lo hace. Se pueden pedir por nombre: dan un error que dice qué falta.
CODIFICADORES_PENDIENTES = {
    "clip_l14": ("openai/clip-vit-large-patch14", "Pareja A, idea 7a (768; misma clase CLIP)"),
    "siglip_base": ("google/siglip-base-patch16-224", "Pareja A, idea 7a (768; AutoModel, sin projection_dim)"),
    "siglip_so400m": ("google/siglip-so400m-patch14-384", "Pareja A, idea 7a, opcional (1152)"),
    "dinov2_base": ("facebook/dinov2-base", "Pareja A, idea 7a, opcional (768; sin texto)"),
}
VARIANTES_PENDIENTES = {
    "relleno": "Pareja A, idea 1: relleno a cuadrado, un vector",
    "3recortes": "Pareja A, idea 1: arriba/centro/abajo, promediar o concatenar",
    "ocr": "Pareja A, idea 2: EasyOCR -> codificar_texto (vector de ceros si no hay texto)",
    "zeroshot": "Pareja B, idea 5: 17 similitudes cartel-frases (clasificador/zeroshot.py)",
}
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def cargar_clip(modelo_hf: str):
    """(procesador, modelo) de CLIP, UNA vez por modelo: si varias variantes o varios miembros
    de un ensamble usan el mismo CLIP, comparten la memoria (la demo corre en una T4)."""
    from transformers import CLIPModel, CLIPProcessor  # aquí, para que las pruebas no lo necesiten

    return CLIPProcessor.from_pretrained(modelo_hf), CLIPModel.from_pretrained(modelo_hf).to(DISPOSITIVO).eval()


def pendiente(codificador: str, variante: str = "centro") -> None:
    """Error claro si se pide algo que todavía no existe."""
    if codificador in CODIFICADORES_PENDIENTES:
        raise NotImplementedError(f"Codificador '{codificador}': pendiente de la "
                                  f"{CODIFICADORES_PENDIENTES[codificador][1]}.")
    if variante in VARIANTES_PENDIENTES:
        raise NotImplementedError(f"Variante '{variante}': pendiente de la {VARIANTES_PENDIENTES[variante]}.")
    if codificador not in CODIFICADORES:
        raise ValueError(f"Codificador desconocido: '{codificador}'.")
    if variante not in VARIANTES:
        raise ValueError(f"Variante desconocida: '{variante}'.")


def recortes(imagen, variante: str = "centro") -> list:
    """Las imágenes que ve el codificador para UN cartel, según la variante (idea 1).

    centro      [cartel]: el procesador de CLIP reescala y recorta el cuadrado central
    relleno     [cartel con bordes hasta ser cuadrado]: CLIP lo ve completo     (PENDIENTE)
    3recortes   [arriba, centro, abajo]: tres cuadrados que cubren el cartel     (PENDIENTE)

    Se usa IGUAL al calcular los vectores y al predecir (la clase del codificador la llama)."""
    if variante == "centro":
        return [imagen]
    pendiente("clip_b32", variante)
    raise ValueError(f"Variante sin recortes: '{variante}'.")


def muestra_ids(ids, cuantos: int) -> set[int]:
    """`cuantos` películas fijas (por hash del id): la misma muestra en cada corrida y cuenta."""
    orden = sorted((int(i) for i in ids), key=lambda i: hashlib.md5(str(i).encode()).hexdigest())
    return set(orden[:cuantos])


class Codificador:
    """Interfaz común: todo codificador tiene `nombre`, `modelo_hf`, `dim` y `codificar`."""

    nombre: str
    modelo_hf: str
    dim: int

    def codificar(self, imagenes) -> np.ndarray:
        """Lista de imágenes PIL -> matriz float32 (n, dim)."""
        raise NotImplementedError


class CodificadorCLIP(Codificador):
    """CLIP de OpenAI. Port de `CodificadorCLIP` del repo principal: vectores normalizados."""

    def __init__(self, codificador: str = "clip_b32", variante: str = "centro"):
        pendiente(codificador, variante)
        self.nombre = f"{codificador}__{variante}"
        self.modelo_hf = CODIFICADORES[codificador]
        self.procesador, self.modelo = cargar_clip(self.modelo_hf)
        self.dim = self.modelo.config.projection_dim

    @torch.inference_mode()
    def codificar(self, imagenes) -> np.ndarray:
        entrada = self.procesador(images=imagenes, return_tensors="pt").to(DISPOSITIVO)
        vectores = self.modelo.get_image_features(**entrada)
        if not torch.is_tensor(vectores):  # transformers >= 5 regresa un objeto
            vectores = vectores.pooler_output
        return torch.nn.functional.normalize(vectores, dim=-1).float().cpu().numpy()

    @torch.inference_mode()
    def codificar_texto(self, textos: list[str]) -> np.ndarray:
        """Vector CLIP de textos (lee los primeros 77 tokens). Lo usarán OCR y zero-shot."""
        entrada = self.procesador(text=textos, return_tensors="pt", padding=True,
                                  truncation=True, max_length=77).to(DISPOSITIVO)
        vectores = self.modelo.get_text_features(**entrada)
        if not torch.is_tensor(vectores):
            vectores = vectores.pooler_output
        return torch.nn.functional.normalize(vectores, dim=-1).float().cpu().numpy()


def crear_codificador(nombre_vectores: str) -> Codificador:
    """`clip_b32__centro` -> el codificador que produce esos vectores."""
    codificador, _, variante = nombre_vectores.partition("__")
    variante = variante or "centro"
    pendiente(codificador, variante)
    if codificador.startswith("clip_"):
        return CodificadorCLIP(codificador, variante)
    raise NotImplementedError(f"Codificador '{codificador}': falta registrarlo en crear_codificador.")


def calcular_vectores(codificador: str, variante: str = "centro", lote: int = 64,
                      muestra: int | None = None) -> None:
    """Vectores de todos los carteles del dataset -> `vectores/<codificador>__<variante>.npz` + `.json`.

    Guarda un pedazo por archivo del dataset en `vectores/partes/<nombre>/`: si Colab
    se desconecta, al volver a correrlo continúa donde se quedó.

    `muestra`: solo esas películas (fijas, de todos los splits), para probar una idea cara
    (p. ej. OCR con 2 mil) antes de correrla en todo el dataset. OJO: un experimento con
    estos vectores solo ve esas películas; compárenlo contra la base en las mismas."""
    modelo = crear_codificador(f"{codificador}__{variante}")
    destino = rutas.carpeta_vectores() / f"{modelo.nombre}.npz"
    partes = rutas.carpeta_vectores() / "partes" / (modelo.nombre + (f"_muestra{muestra}" if muestra else ""))
    elegidas = muestra_ids(datos.metadata()["id"], muestra) if muestra else None
    partes.mkdir(parents=True, exist_ok=True)
    archivos = datos.archivos()
    for numero, archivo in enumerate(archivos, start=1):
        salida = partes / f"{archivo.stem}.npz"
        if salida.exists() and salida.stat().st_mtime > archivo.stat().st_mtime:
            continue
        filas = pq.read_table(archivo, columns=["id", "imagen"]).to_pylist()
        if elegidas is not None:
            filas = [f for f in filas if int(f["id"]) in elegidas]
        if not filas:
            np.savez(salida, ids=np.array([], dtype=np.int64), vectores=np.zeros((0, modelo.dim), np.float32))
            continue
        vectores = [modelo.codificar([datos.abrir_cartel(f["imagen"]) for f in filas[i:i + lote]])
                    for i in range(0, len(filas), lote)]
        np.savez(salida, ids=np.array([f["id"] for f in filas]), vectores=np.concatenate(vectores))
        print(f"{numero}/{len(archivos)} archivos ({len(filas)} carteles)", flush=True)

    guardados = [np.load(partes / f"{a.stem}.npz") for a in archivos]
    ids = np.concatenate([g["ids"] for g in guardados])
    np.savez(destino, ids=ids, vectores=np.concatenate([g["vectores"] for g in guardados]).astype(np.float32))
    destino.with_suffix(".json").write_text(json.dumps({
        "nombre": modelo.nombre, "codificador": codificador, "variante": variante,
        "modelo_hf": modelo.modelo_hf, "dim": modelo.dim, "peliculas": int(len(ids)), "muestra": muestra,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False))
    print(f"Guardados {len(ids)} vectores en {destino}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("codificador", choices=sorted({**CODIFICADORES, **CODIFICADORES_PENDIENTES}))
    parser.add_argument("--variante", default="centro", choices=VARIANTES + list(VARIANTES_PENDIENTES))
    parser.add_argument("--lote", type=int, default=64)
    parser.add_argument("--muestra", type=int, default=None, help="solo N películas fijas (prueba chica)")
    args = parser.parse_args()
    calcular_vectores(args.codificador, args.variante, args.lote, args.muestra)
