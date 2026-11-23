"""Creación de «Fotos Minis.app» (estructura del paquete de macOS)."""

import plistlib
import subprocess

from PIL import Image

from fotos_minis.instalar_app import EJECUTABLE, NOMBRE, crear_app


def test_estructura_del_paquete(tmp_path):
    proyecto = tmp_path / "Mis cosas" / "Fotos-minis d'Carlos"     # espacios y apóstrofo
    (proyecto / "recursos").mkdir(parents=True)
    app = crear_app(tmp_path / "Aplicaciones", proyecto)
    assert app.name == f"{NOMBRE}.app"

    info = plistlib.load(open(app / "Contents" / "Info.plist", "rb"))
    assert info["CFBundleExecutable"] == EJECUTABLE and info["CFBundlePackageType"] == "APPL"

    lanzador = app / "Contents" / "MacOS" / EJECUTABLE
    assert lanzador.stat().st_mode & 0o111                        # ejecutable
    subprocess.run(["bash", "-n", str(lanzador)], check=True)      # sintaxis válida
    # La ruta del proyecto se interpreta bien en bash aunque tenga espacios y comillas.
    linea = next(l for l in lanzador.read_text().splitlines() if l.startswith("PROYECTO="))
    eco = subprocess.run(["bash", "-c", f'{linea}; printf %s "$PROYECTO"'], capture_output=True, text=True)
    assert eco.stdout == str(proyecto)

    icono = Image.open(app / "Contents" / "Resources" / "icono.icns")
    assert (512, 512, 2) in icono.info["sizes"]


def test_reinstalar_sustituye_la_app(tmp_path):
    proyecto = tmp_path / "proyecto"
    (proyecto / "recursos").mkdir(parents=True)
    app = crear_app(tmp_path, proyecto)
    (app / "Contents" / "basura.txt").write_text("de la versión anterior")
    app = crear_app(tmp_path, proyecto)
    assert not (app / "Contents" / "basura.txt").exists()
