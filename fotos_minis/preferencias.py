"""Preferencias de la aplicación (p. ej. la apariencia), junto a los perfiles."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

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


def _escribir(clave: str, valor) -> None:
    datos = _leer()
    datos[clave] = valor
    p = ruta()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def fijar_apariencia(valor: str) -> None:
    if valor not in APARIENCIAS:
        raise ValueError(f"Apariencia no válida: {valor}")
    _escribir("apariencia", valor)


# --------------------------------------------------------------------------- #
# Carpeta de guardado
# --------------------------------------------------------------------------- #

def carpeta_por_defecto() -> Path:
    return Path.cwd() / "fotos"


def carpeta_guardado() -> Path:
    """Carpeta donde se guardan los resultados: la última elegida por el usuario
    (o «fotos» junto a donde se ejecuta la aplicación). Se crea si no existe; si la
    guardada ya no es utilizable (p. ej. un disco externo desconectado) se usa la
    carpeta por defecto."""
    if os.environ.get("FOTOS_MINIS_SALIDA"):
        candidatas = [Path(os.environ["FOTOS_MINIS_SALIDA"])]
    else:
        guardada = _leer().get("carpeta_guardado")
        candidatas = ([Path(guardada)] if guardada else []) + [carpeta_por_defecto()]
    for c in candidatas:
        try:
            c.mkdir(parents=True, exist_ok=True)
            if os.access(c, os.W_OK):
                return c
        except OSError:
            continue
    raise OSError(f"No se puede escribir en {candidatas[-1]}")


def fijar_carpeta_guardado(texto: str) -> Path:
    """Valida, crea si hace falta y recuerda la carpeta. ValueError si no sirve."""
    texto = (texto or "").strip().strip('"').strip("'")
    if not texto:
        raise ValueError("Indica una carpeta.")
    carpeta = Path(texto).expanduser()
    if not carpeta.is_absolute():
        carpeta = Path.cwd() / carpeta
    carpeta = carpeta.resolve()
    if carpeta.exists() and not carpeta.is_dir():
        raise ValueError(f"{carpeta} es un fichero, no una carpeta.")
    try:
        carpeta.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(f"No se puede crear la carpeta {carpeta}: {e.strerror or e}") from e
    if not carpeta.is_dir():
        raise ValueError(f"{carpeta} no es una carpeta.")
    if not os.access(carpeta, os.W_OK):
        raise ValueError(f"No hay permiso para guardar en {carpeta}.")
    _escribir("carpeta_guardado", str(carpeta))
    return carpeta


def elegir_carpeta_sistema(inicial: Path | None = None) -> str | None:
    """Abre el selector de carpetas del sistema (la app se ejecuta en el propio
    ordenador). Devuelve la ruta elegida, o None si se cancela o no hay selector."""
    inicial = inicial if inicial and inicial.is_dir() else Path.home()
    sistema = platform.system()
    try:
        if sistema == "Darwin":
            guion = ['activate',
                     f'set f to choose folder with prompt "Carpeta donde guardar las fotos" '
                     f'default location (POSIX file "{inicial}")',
                     'POSIX path of f']
            args = ["osascript"] + [x for linea in guion for x in ("-e", linea)]
        elif sistema == "Linux" and shutil.which("zenity"):
            args = ["zenity", "--file-selection", "--directory", f"--filename={inicial}/"]
        elif sistema == "Windows":
            ps = ("Add-Type -AssemblyName System.Windows.Forms;"
                  "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
                  f"$d.SelectedPath='{inicial}';"
                  "if($d.ShowDialog() -eq 'OK'){$d.SelectedPath}")
            args = ["powershell", "-NoProfile", "-Command", ps]
        else:
            return None
        r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    ruta_elegida = r.stdout.strip()
    return ruta_elegida if r.returncode == 0 and ruta_elegida else None


def abrir_en_explorador(carpeta: Path) -> bool:
    """Muestra la carpeta en el Finder / Explorador."""
    sistema = platform.system()
    orden = {"Darwin": ["open"], "Windows": ["explorer"], "Linux": ["xdg-open"]}.get(sistema)
    if not orden or (sistema == "Linux" and not shutil.which("xdg-open")):
        return False
    try:
        subprocess.Popen(orden + [str(carpeta)])
        return True
    except OSError:
        return False
