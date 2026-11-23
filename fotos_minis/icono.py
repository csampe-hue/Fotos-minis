"""Icono de la aplicación: una miniatura sobre su peana en un estudio negro."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LADO = 1024
SS = 4  # sobremuestreo para bordes suaves


def _lienzo() -> Image.Image:
    n = LADO * SS
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    # Fondo de estudio: negro con un foco cálido arriba en el centro.
    foco = np.exp(-(((xx - 0.5) / 0.32) ** 2 + ((yy - 0.42) / 0.36) ** 2))
    base = 0.035 + 0.16 * foco
    rgb = np.stack([base * 1.05, base * 1.0, base * 0.95], axis=2)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
    # Forma de icono de macOS: cuadrado redondeado con margen.
    mascara = Image.new("L", (n, n), 0)
    m = 100 * SS
    ImageDraw.Draw(mascara).rounded_rectangle((m, m, n - m, n - m), radius=185 * SS, fill=255)
    img.putalpha(mascara)
    return img


def generar() -> Image.Image:
    img = _lienzo()
    n = LADO * SS
    s = SS
    capa = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    cx = 512 * s

    # Sombra de contacto bajo la peana.
    sombra = Image.new("L", (n, n), 0)
    ImageDraw.Draw(sombra).ellipse((cx - 260 * s, 735 * s, cx + 260 * s, 815 * s), fill=170)
    sombra = sombra.filter(ImageFilter.GaussianBlur(18 * s))
    negro = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    negro.putalpha(sombra)
    img.alpha_composite(negro)

    # Peana: canto oscuro y superficie de arena.
    d.rounded_rectangle((cx - 235 * s, 690 * s, cx + 235 * s, 760 * s), radius=10 * s, fill=(28, 28, 30, 255))
    d.ellipse((cx - 235 * s, 722 * s, cx + 235 * s, 798 * s), fill=(28, 28, 30, 255))
    d.ellipse((cx - 235 * s, 650 * s, cx + 235 * s, 730 * s), fill=(196, 160, 104, 255))
    d.ellipse((cx - 200 * s, 662 * s, cx + 190 * s, 716 * s), fill=(214, 180, 122, 255))

    # Figura: silueta de soldado de pie, en tono bronce.
    bronce = (176, 128, 72, 255)
    claro = (222, 176, 112, 255)
    d.ellipse((cx - 44 * s, 250 * s, cx + 44 * s, 338 * s), fill=bronce)               # cabeza
    d.pieslice((cx - 54 * s, 236 * s, cx + 54 * s, 318 * s), 180, 360, fill=claro)      # casco
    d.polygon([(cx - 92 * s, 352 * s), (cx + 92 * s, 352 * s), (cx + 74 * s, 520 * s),
               (cx - 74 * s, 520 * s)], fill=bronce)                                     # torso
    d.rounded_rectangle((cx - 70 * s, 340 * s, cx + 70 * s, 372 * s), radius=14 * s, fill=bronce)
    d.polygon([(cx - 74 * s, 515 * s), (cx - 8 * s, 515 * s), (cx - 22 * s, 688 * s),
               (cx - 72 * s, 688 * s)], fill=bronce)                                     # pierna izq.
    d.polygon([(cx + 8 * s, 515 * s), (cx + 74 * s, 515 * s), (cx + 86 * s, 688 * s),
               (cx + 32 * s, 688 * s)], fill=bronce)                                     # pierna der.
    d.polygon([(cx - 92 * s, 360 * s), (cx - 60 * s, 360 * s), (cx - 40 * s, 470 * s),
               (cx - 70 * s, 480 * s)], fill=claro)                                      # brazo
    d.line([(cx - 150 * s, 430 * s), (cx + 150 * s, 380 * s)], fill=(60, 52, 44, 255), width=20 * s)  # fusil
    d.polygon([(cx + 92 * s, 360 * s), (cx + 60 * s, 360 * s), (cx + 30 * s, 420 * s),
               (cx + 60 * s, 430 * s)], fill=claro)                                      # brazo der.
    img.alpha_composite(capa)

    # Recorta todo a la forma del icono y reduce con antialias.
    forma = _lienzo().getchannel("A")
    img.putalpha(Image.fromarray(np.minimum(np.asarray(img.getchannel("A")), np.asarray(forma))))
    return img.resize((LADO, LADO), Image.LANCZOS)


def guardar(png: Path, icns: Path | None = None) -> None:
    im = generar()
    png.parent.mkdir(parents=True, exist_ok=True)
    im.save(png)
    if icns is not None:
        icns.parent.mkdir(parents=True, exist_ok=True)
        im.save(icns)  # Pillow genera todos los tamaños del .icns


if __name__ == "__main__":
    guardar(Path("recursos/icono.png"), Path("recursos/icono.icns"))
