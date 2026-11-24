"""Perfiles de ajustes guardados por el usuario.

Se guardan en ``~/.fotos-minis/perfiles.json`` (fuera de la carpeta del programa,
para que sobrevivan a las actualizaciones). Uno de ellos puede marcarse como
predeterminado: es el que se aplica al abrir la aplicación.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields
from pathlib import Path

from .composicion import OpcionesEstudio
from .segmentacion import MODELO_POR_DEFECTO, MODELOS

FABRICA = "Valores de fábrica"

# Ajustes de estudio que forman parte de un perfil (en el orden de la interfaz).
CAMPOS_ESTUDIO = [
    "halo", "vineteado", "sombra", "reflejo", "reflejo_caida", "oscurecer_borde",
    "corregir_dominante", "negro", "reencuadrar", "margen", "proporcion",
    "descontaminar_bordes", "suelo", "suelo_ajuste", "horizonte", "horizonte_distancia", "horizonte_desenfoque",
    "horizonte_suelo", "horizonte_altura", "carrusel",
]
CAMPOS = ["modelo", "rellenar_huecos", *CAMPOS_ESTUDIO]


def ruta() -> Path:
    return Path(os.environ.get("FOTOS_MINIS_PERFILES", Path.home() / ".fotos-minis" / "perfiles.json"))


def valores_fabrica() -> dict:
    op = asdict(OpcionesEstudio())
    return {"modelo": MODELO_POR_DEFECTO, "rellenar_huecos": 0,
            **{k: op[k] for k in CAMPOS_ESTUDIO}}


def _leer() -> dict:
    p = ruta()
    try:
        datos = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(datos.get("perfiles"), dict):
            raise ValueError
        return datos
    except FileNotFoundError:
        return {"predeterminado": None, "perfiles": {}}
    except (ValueError, OSError):
        # Fichero dañado: se aparta para no perderlo y se empieza de cero.
        try:
            p.replace(p.with_suffix(".json.dañado"))
        except OSError:
            pass
        return {"predeterminado": None, "perfiles": {}}


def _escribir(datos: dict) -> None:
    p = ruta()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)  # escritura atómica


def _sanear(valores: dict) -> dict:
    """Completa con valores de fábrica lo que falte (perfiles de versiones anteriores)
    y descarta claves desconocidas o valores no válidos."""
    base = valores_fabrica()
    for k in CAMPOS:
        if k in valores and valores[k] is not None:
            base[k] = valores[k]
    if base["modelo"] not in MODELOS:
        base["modelo"] = MODELO_POR_DEFECTO
    from .composicion import SUELOS
    if base["suelo"] not in SUELOS:
        base["suelo"] = "estudio"
    try:
        base["carrusel"] = int(base["carrusel"]) if int(base["carrusel"]) in (-1, 2, 3, 4) else 0
    except (TypeError, ValueError):
        base["carrusel"] = 0
    return base


def listar() -> list[str]:
    return sorted(_leer()["perfiles"], key=str.casefold)


def obtener(nombre: str | None) -> dict:
    if not nombre or nombre == FABRICA:
        return valores_fabrica()
    perfiles = _leer()["perfiles"]
    if nombre not in perfiles:
        raise KeyError(f"No existe el perfil «{nombre}»")
    return _sanear(perfiles[nombre])


def guardar(nombre: str, valores: dict, predeterminado: bool = False) -> str:
    nombre = " ".join(str(nombre).split())
    if not nombre:
        raise ValueError("Escribe un nombre para el perfil.")
    if nombre == FABRICA:
        raise ValueError(f"«{FABRICA}» es un nombre reservado.")
    datos = _leer()
    datos["perfiles"][nombre] = _sanear(valores)
    if predeterminado:
        datos["predeterminado"] = nombre
    _escribir(datos)
    return nombre


def borrar(nombre: str) -> None:
    datos = _leer()
    datos["perfiles"].pop(nombre, None)
    if datos.get("predeterminado") == nombre:
        datos["predeterminado"] = None
    _escribir(datos)


def predeterminado() -> str:
    """Nombre del perfil que se aplica al abrir (o «Valores de fábrica»)."""
    datos = _leer()
    nombre = datos.get("predeterminado")
    return nombre if nombre in datos["perfiles"] else FABRICA


def fijar_predeterminado(nombre: str | None) -> None:
    datos = _leer()
    datos["predeterminado"] = None if (not nombre or nombre == FABRICA) else nombre
    _escribir(datos)


def a_opciones(valores: dict) -> tuple[OpcionesEstudio, str, int]:
    """Convierte un perfil en (opciones de estudio, modelo, px de huecos a rellenar)."""
    v = _sanear(valores)
    validos = {f.name for f in fields(OpcionesEstudio)}
    op = OpcionesEstudio(**{k: v[k] for k in CAMPOS_ESTUDIO if k in validos})
    op.negro = int(op.negro)
    return op, v["modelo"], int(v["rellenar_huecos"])
