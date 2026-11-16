"""Preferencias de la aplicación (p. ej. la apariencia), junto a los perfiles."""

from __future__ import annotations

import json

from . import perfiles

APARIENCIAS = {"sistema": "Sistema", "diurno": "Diurno", "nocturno": "Nocturno"}


def ruta():
    return perfiles.ruta().with_name("preferencias.json")


def _leer() -> dict:
    try:
        datos = json.loads(ruta().read_text(encoding="utf-8"))
        return datos if isinstance(datos, dict) else {}
    except (OSError, ValueError):
        return {}


def apariencia() -> str:
    valor = _leer().get("apariencia", "sistema")
    return valor if valor in APARIENCIAS else "sistema"


def fijar_apariencia(valor: str) -> None:
    if valor not in APARIENCIAS:
        raise ValueError(f"Apariencia no válida: {valor}")
    datos = _leer()
    datos["apariencia"] = valor
    p = ruta()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
