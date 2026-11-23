#!/bin/bash
# Crea «Fotos Minis.app» en Aplicaciones: doble clic en Finder.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

fallo() { echo; echo "ERROR: $1"; echo "Pulsa Intro para cerrar esta ventana."; read -r; exit 1; }

echo "=== Instalar Fotos Minis en Aplicaciones ==="
if ! command -v uv >/dev/null 2>&1; then
    echo "Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || fallo "no se pudo instalar uv."
    export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -x ".venv/bin/python" ]; then
    echo "Preparando Python 3.12..."
    uv venv --python 3.12 .venv || fallo "no se pudo crear el entorno de Python 3.12."
fi
echo "Instalando o actualizando las librerías (incluida la ventana propia)..."
VIRTUAL_ENV="$PWD/.venv" uv pip install -r requirements.txt || fallo "no se pudieron instalar las librerías."

echo
echo "Icono de la app:"
echo "  · Para usar una imagen tuya, arrastra aquí el archivo (JPG, PNG o HEIC) y pulsa Intro."
echo "  · Escribe «original» para volver al icono por defecto."
echo "  · Pulsa Intro sin más para dejar el que tenga."
read -r -p "> " ICONO
if [ "$ICONO" = "original" ]; then
    ARGS=(--icono-predeterminado)
elif [ -n "$ICONO" ]; then
    ARGS=(--icono "$ICONO")
else
    ARGS=()
fi
.venv/bin/python -m fotos_minis.instalar_app "${ARGS[@]}" || fallo "no se pudo crear la aplicación."
APP="/Applications/Fotos Minis.app"; [ -d "$APP" ] || APP="$HOME/Applications/Fotos Minis.app"
echo
echo "Listo: ya tienes «Fotos Minis» en Aplicaciones (y en Launchpad)."
echo "Para tenerla siempre a mano, arrástrala al Dock."
open -R "$APP" 2>/dev/null
echo "Pulsa Intro para cerrar esta ventana."
read -r
