"""Clasificador de géneros de Poster2Movie (cartel -> géneros), versión del equipo.

El repo principal vive como submódulo en `poster2movie/`. De ahí se usan
`p2m.config` (los 17 géneros, el split por franquicia, las carpetas) y
`p2m.datos` (lectura del dataset), sin copiarlos: así el split de test es
el mismo por construcción.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SUBMODULO = RAIZ / "poster2movie"

if not (SUBMODULO / "p2m" / "config.py").exists():
    raise ImportError(
        "Falta el submódulo del repo principal en 'poster2movie/'. "
        "Clona con `git clone --recurse-submodules ...` o corre `git submodule update --init`."
    )
if str(SUBMODULO) not in sys.path:
    sys.path.insert(0, str(SUBMODULO))
