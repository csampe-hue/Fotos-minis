"""Crea «Fotos Minis.app» en Aplicaciones (macOS).

La aplicación es un envoltorio ligero: al abrirla ejecuta ``escritorio.py`` con el
Python del entorno ``.venv`` de esta carpeta, sin ventana de Terminal, y muestra la
interfaz en una ventana propia. Si se mueve la carpeta del programa, basta con volver
a ejecutar el instalador.

Uso:  .venv/bin/python -m fotos_minis.instalar_app [--destino CARPETA]
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import stat
from pathlib import Path

from .icono import guardar as guardar_icono

NOMBRE = "Fotos Minis"
EJECUTABLE = "FotosMinis"
PROYECTO = Path(__file__).resolve().parent.parent

GUION = r'''#!/bin/bash
# Lanzador de «Fotos Minis.app» (generado por fotos_minis/instalar_app.py).
PROYECTO={proyecto}
LOG="$HOME/.fotos-minis/app.log"
mkdir -p "$HOME/.fotos-minis"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

aviso() {{
    osascript -e "display alert \"Fotos Minis\" message \"$1\" as critical" >/dev/null 2>&1
}}

cd "$PROYECTO" 2>/dev/null || {{
    aviso "No se encuentra la carpeta del programa: $PROYECTO. Si la has movido, vuelve a ejecutar «Instalar app en Aplicaciones.command» desde su nuevo sitio."
    exit 1
}}

{{
    echo "=== $(date) ==="
    # Primera vez o entorno incompleto: se prepara (Python 3.12 + librerías).
    if [ ! -x ".venv/bin/python" ] || ! .venv/bin/python -c "import gradio, webview" 2>/dev/null; then
        osascript -e 'display notification "Preparando la aplicación (sólo la primera vez, puede tardar unos minutos)…" with title "Fotos Minis"'
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh || exit 10
            export PATH="$HOME/.local/bin:$PATH"
        fi
        [ -x ".venv/bin/python" ] || uv venv --python 3.12 .venv || exit 11
        VIRTUAL_ENV="$PWD/.venv" uv pip install -r requirements.txt || exit 12
    fi
}} >>"$LOG" 2>&1 || {{
    aviso "No se pudo preparar la aplicación. Detalles en $LOG"
    exit 1
}}

exec .venv/bin/python escritorio.py >>"$LOG" 2>&1
'''


def _plist() -> dict:
    return {
        "CFBundleName": NOMBRE,
        "CFBundleDisplayName": NOMBRE,
        "CFBundleIdentifier": "local.fotos-minis.app",
        "CFBundleExecutable": EJECUTABLE,
        "CFBundleIconFile": "icono",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "11.0",
        "LSApplicationCategoryType": "public.app-category.photography",
        "NSHighResolutionCapable": True,
        "NSAppleEventsUsageDescription": "Para mostrar el selector de carpetas de guardado.",
    }


def _citar(ruta: Path) -> str:
    """Ruta entre comillas simples para bash (admite espacios y acentos)."""
    return "'" + str(ruta).replace("'", "'\\''") + "'"


def crear_app(destino: Path, proyecto: Path = PROYECTO) -> Path:
    app = destino / f"{NOMBRE}.app"
    if app.exists():
        shutil.rmtree(app)
    contenido = app / "Contents"
    (contenido / "MacOS").mkdir(parents=True)
    (contenido / "Resources").mkdir()
    with open(contenido / "Info.plist", "wb") as f:
        plistlib.dump(_plist(), f)
    ejecutable = contenido / "MacOS" / EJECUTABLE
    ejecutable.write_text(GUION.format(proyecto=_citar(proyecto)), encoding="utf-8")
    ejecutable.chmod(ejecutable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    guardar_icono(proyecto / "recursos" / "icono.png", contenido / "Resources" / "icono.icns")
    return app


def destino_por_defecto() -> Path:
    """/Applications si se puede escribir en ella; si no, ~/Applications."""
    sistema = Path("/Applications")
    if sistema.is_dir() and os.access(sistema, os.W_OK):
        return sistema
    personal = Path.home() / "Applications"
    personal.mkdir(exist_ok=True)
    return personal


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--destino", type=Path, help="carpeta donde crear la app (por defecto, Aplicaciones)")
    a = p.parse_args(argv)
    app = crear_app(a.destino or destino_por_defecto())
    print(f"Creada: {app}")


if __name__ == "__main__":
    main()
