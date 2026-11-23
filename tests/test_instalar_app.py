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


def _imagen(ruta, color=(30, 120, 200), w=600, h=400):
    from PIL import Image as I
    im = I.new("RGB", (w, h), (0, 0, 0))
    im.paste(I.new("RGB", (h, h), color), ((w - h) // 2, 0))   # cuadrado central de color
    im.save(ruta)
    return ruta


def test_icono_propio_recortado_redondeado_y_recordado(tmp_path):
    from fotos_minis.instalar_app import icono_en_uso, icono_personal
    proyecto = tmp_path / "proyecto"
    (proyecto / "recursos").mkdir(parents=True)
    foto = _imagen(tmp_path / "mi mini.png")
    app = crear_app(tmp_path / "Apps", proyecto, icono=foto)
    icono = Image.open(icono_en_uso(proyecto)).convert("RGBA")
    assert icono.size == (1024, 1024)
    assert icono.getpixel((512, 512))[:3] == (30, 120, 200)        # centro de la foto
    assert icono.getpixel((5, 5))[3] == 0                          # margen transparente
    assert icono.getpixel((112, 112))[3] < 255                      # esquina redondeada
    assert icono_personal(proyecto).exists()
    # Reinstalar sin indicar icono conserva el propio.
    crear_app(tmp_path / "Apps", proyecto)
    assert Image.open(icono_en_uso(proyecto)).convert("RGB").getpixel((512, 512)) == (30, 120, 200)
    # «original» vuelve al predeterminado.
    crear_app(tmp_path / "Apps", proyecto, icono_predeterminado=True)
    assert not icono_personal(proyecto).exists()
    assert Image.open(icono_en_uso(proyecto)).convert("RGB").getpixel((512, 512)) != (30, 120, 200)


def test_ruta_arrastrada_desde_terminal():
    from fotos_minis.instalar_app import limpiar_ruta
    assert str(limpiar_ruta(r"/Users/yo/Mis\ fotos/mini\ (1).jpg ")) == "/Users/yo/Mis fotos/mini (1).jpg"
    assert str(limpiar_ruta("'/Users/yo/Mis fotos/mini.heic'")) == "/Users/yo/Mis fotos/mini.heic"
