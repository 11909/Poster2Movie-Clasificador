"""Prepara un release: mide en val Y en test, y junta todo en `releases/<tag>/`.

    python -m clasificador.release v0.1.0 base --notas "Base reproducida del repo principal"

Copia a `releases/<tag>/`:
- `<nombre>.pt` (la cabeza: pesa de KB a pocos MB, por eso sí va en git)
- `metricas_val.json`, `metricas_test.json`, `por_genero_val.csv`, `por_genero_test.csv`
- `<nombre>.yaml` (la receta del experimento)
- `LEEME.md`: resumen y comparación contra el release anterior. Es el texto del release en GitHub.

Después: commit en la rama `release-<tag>` y Pull Request; al fusionarlo, `git tag <tag>` en
`main` y `git push origin <tag>`. GitHub Actions crea el release (notebook `C_release`).
"""

import argparse
import json
import re
import shutil

import pandas as pd

from clasificador import RAIZ, rutas
from clasificador.datos import GENEROS_DEBILES
from clasificador.evaluar import evaluar
from p2m import config


def release_anterior(tag: str) -> str | None:
    def version(nombre: str) -> tuple[int, ...]:
        return tuple(int(n) for n in nombre.lstrip("v").split("."))

    anteriores = [c.name for c in rutas.RELEASES.glob("v*")
                  if c.is_dir() and re.fullmatch(r"v\d+\.\d+\.\d+", c.name) and version(c.name) < version(tag)]
    return max(anteriores, key=version) if anteriores else None


def preparar(tag: str, nombre: str, notas: str = "") -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise SystemExit("El tag debe ser vMAYOR.MENOR.PARCHE, por ejemplo v0.2.0.")
    destino = rutas.RELEASES / tag
    if destino.exists():
        raise SystemExit(f"{destino} ya existe: un release no se reescribe, usa otro número.")

    val = evaluar(nombre, "val")
    test = evaluar(nombre, "test", para_release=True)

    destino.mkdir(parents=True)
    shutil.copy(rutas.ruta_modelo(nombre), destino / f"{nombre}.pt")
    for split in ("val", "test"):
        for archivo in (f"metricas_{split}.json", f"por_genero_{split}.csv"):
            shutil.copy(rutas.RESULTADOS / val["nombre"] / archivo, destino / archivo)
    receta = RAIZ / "experimentos" / f"{nombre}.yaml"
    if receta.exists():
        shutil.copy(receta, destino / receta.name)

    anterior = release_anterior(tag)
    lineas = [f"# {tag} — {nombre}", "", notas or "(sin notas)", "",
              f"Vectores: {', '.join(val['vectores'])} · commit del modelo: {val['commit_modelo']}", "",
              "| Métrica | Validación | Test |", "| --- | --- | --- |"]
    for clave in ("F1_macro", "F1_micro", "mAP", "acierto_exacto"):
        lineas.append(f"| {clave} | {val[clave]:.3f} | {test[clave]:.3f} |")
    for g in GENEROS_DEBILES:
        lineas.append(f"| F1 {config.GENEROS_ES[g]} | {val['F1_debiles'][g]:.3f} | {test['F1_debiles'][g]:.3f} |")

    if anterior:
        previo = pd.read_csv(rutas.RELEASES / anterior / "por_genero_test.csv").set_index("genero")["F1"]
        actual = pd.read_csv(destino / "por_genero_test.csv").set_index("genero")
        lineas += ["", f"## Contra {anterior} (F1 en test)", "",
                   f"| Género | {anterior} | {tag} | Diferencia |", "| --- | --- | --- | --- |"]
        orden = GENEROS_DEBILES + [g for g in actual.index if g not in GENEROS_DEBILES]
        for g in orden:
            lineas.append(f"| {config.GENEROS_ES[g]} | {previo[g]:.3f} | {actual.loc[g, 'F1']:.3f} | "
                          f"{actual.loc[g, 'F1'] - previo[g]:+.3f} |")
        macro_previo = json.loads((rutas.RELEASES / anterior / "metricas_test.json").read_text())["F1_macro"]
        lineas.append(f"| **F1 macro** | {macro_previo:.3f} | {test['F1_macro']:.3f} | "
                      f"{test['F1_macro'] - macro_previo:+.3f} |")

    (destino / "LEEME.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\nListo: {destino}\nAhora (o con el notebook C_release):\n"
          f"  git checkout -b release-{tag}\n  git add releases/{tag} resultados\n"
          f"  git commit -m \"Release {tag}\"\n  git push origin release-{tag}   # y abrir el Pull Request\n"
          f"Al fusionarlo, en main:\n  git tag {tag} && git push origin {tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="vMAYOR.MENOR.PARCHE")
    parser.add_argument("nombre", help="experimento a publicar (su .pt en <P2M_SALIDAS>/modelos/)")
    parser.add_argument("--notas", default="", help="qué cambió y por qué")
    args = parser.parse_args()
    preparar(args.tag, args.nombre, args.notas)
