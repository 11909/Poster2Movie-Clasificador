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
"""

import argparse
import json
from datetime import datetime, timezone

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
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


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
        if variante != "centro":
            raise NotImplementedError(f"Variante '{variante}': pendiente de la Pareja A.")
        from transformers import CLIPModel, CLIPProcessor  # aquí, para que las pruebas no lo necesiten

        self.nombre = f"{codificador}__{variante}"
        self.modelo_hf = CODIFICADORES[codificador]
        self.procesador = CLIPProcessor.from_pretrained(self.modelo_hf)
        self.modelo = CLIPModel.from_pretrained(self.modelo_hf).to(DISPOSITIVO).eval()
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
    if codificador.startswith("clip_"):
        return CodificadorCLIP(codificador, variante or "centro")
    raise NotImplementedError(f"Codificador '{codificador}': pendiente de la Pareja A.")


def calcular_vectores(codificador: str, variante: str = "centro", lote: int = 64) -> None:
    """Vectores de todos los carteles del dataset -> `vectores/<codificador>__<variante>.npz` + `.json`.

    Guarda un pedazo por archivo del dataset en `vectores/partes/<nombre>/`: si Colab
    se desconecta, al volver a correrlo continúa donde se quedó."""
    modelo = crear_codificador(f"{codificador}__{variante}")
    destino = rutas.carpeta_vectores() / f"{modelo.nombre}.npz"
    partes = rutas.carpeta_vectores() / "partes" / modelo.nombre
    partes.mkdir(parents=True, exist_ok=True)
    archivos = datos.archivos()
    for numero, archivo in enumerate(archivos, start=1):
        salida = partes / f"{archivo.stem}.npz"
        if salida.exists() and salida.stat().st_mtime > archivo.stat().st_mtime:
            continue
        filas = pq.read_table(archivo, columns=["id", "imagen"]).to_pylist()
        vectores = [modelo.codificar([datos.abrir_cartel(f["imagen"]) for f in filas[i:i + lote]])
                    for i in range(0, len(filas), lote)]
        np.savez(salida, ids=np.array([f["id"] for f in filas]), vectores=np.concatenate(vectores))
        print(f"{numero}/{len(archivos)} archivos ({len(filas)} carteles)", flush=True)

    guardados = [np.load(partes / f"{a.stem}.npz") for a in archivos]
    ids = np.concatenate([g["ids"] for g in guardados])
    np.savez(destino, ids=ids, vectores=np.concatenate([g["vectores"] for g in guardados]).astype(np.float32))
    destino.with_suffix(".json").write_text(json.dumps({
        "nombre": modelo.nombre, "codificador": codificador, "variante": variante,
        "modelo_hf": modelo.modelo_hf, "dim": modelo.dim, "peliculas": int(len(ids)),
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False))
    print(f"Guardados {len(ids)} vectores en {destino}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("codificador", choices=sorted(CODIFICADORES))
    parser.add_argument("--variante", default="centro", choices=VARIANTES)
    parser.add_argument("--lote", type=int, default=64)
    args = parser.parse_args()
    calcular_vectores(args.codificador, args.variante, args.lote)
