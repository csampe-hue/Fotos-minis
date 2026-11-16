#!/bin/bash
# Fotos Minis: doble clic en Finder para arrancar la aplicación y abrir el navegador.
# La primera vez prepara el entorno (Python 3.12 + librerías); después arranca directo.

URL="http://127.0.0.1:7860"
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

abrir_navegador() {
    if command -v open >/dev/null 2>&1; then open "$URL"; else xdg-open "$URL" >/dev/null 2>&1; fi
}

fallo() {
    echo
    echo "ERROR: $1"
    echo "Pulsa Intro para cerrar esta ventana."
    read -r
    exit 1
}

echo "=== Fotos Minis · Fondo negro de estudio ==="

# ¿Ya está en marcha? Entonces sólo se abre el navegador.
if curl -s -o /dev/null "$URL"; then
    echo "La aplicación ya estaba abierta. Abriendo el navegador..."
    abrir_navegador
    exit 0
fi

# Primera vez: crear el entorno con Python 3.12.
if [ ! -x ".venv/bin/python" ]; then
    echo "Primera ejecución: preparando el entorno (puede tardar unos minutos)..."
    if ! command -v uv >/dev/null 2>&1; then
        echo "Instalando uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || fallo "no se pudo instalar uv."
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv venv --python 3.12 .venv || fallo "no se pudo crear el entorno de Python 3.12."
    VIRTUAL_ENV="$PWD/.venv" uv pip install -r requirements.txt \
        || { rm -rf .venv; fallo "no se pudieron instalar las librerías."; }
fi

# Arrancar la aplicación en segundo plano y cerrarla al cerrar esta ventana.
.venv/bin/python app.py &
PID=$!
trap 'echo; echo "Cerrando Fotos Minis..."; kill $PID 2>/dev/null; wait $PID 2>/dev/null; exit 0' INT TERM HUP

echo "Arrancando..."
for _ in $(seq 1 180); do
    if curl -s -o /dev/null "$URL"; then
        echo
        echo "Lista en $URL"
        echo "Para cerrar la aplicación, cierra esta ventana o pulsa Ctrl + C."
        abrir_navegador
        break
    fi
    kill -0 $PID 2>/dev/null || fallo "la aplicación se cerró al arrancar (mira los mensajes de arriba)."
    sleep 1
done

wait $PID
echo
echo "Fotos Minis se ha cerrado. Ya puedes cerrar esta ventana."
