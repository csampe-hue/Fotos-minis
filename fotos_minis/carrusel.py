"""Carrusel de Instagram: una imagen N:1 dividida en N cuadrados contiguos.

Publicadas como carrusel, al deslizar se ven como una única imagen continua.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

PARTES = (2, 3, 4)


def dividir(imagen: Image.Image, n: int) -> list[Image.Image]:
    """Corta la imagen en n cuadrados de izquierda a derecha, sin perder ni repetir
    ningún píxel (el ancho debe ser exactamente n veces el alto)."""
    w, h = imagen.size
    if n not in PARTES or w != n * h:
        raise ValueError(f"Para {n} partes la imagen debe medir {n}×alto de ancho (mide {w}×{h}).")
    partes = []
    for i in range(n):
        parte = imagen.crop((i * h, 0, (i + 1) * h, h))
        parte.info.update(imagen.info)  # conserva el perfil de color
        partes.append(parte)
    return partes


def nombre_parte(base: str, i: int, n: int, ext: str = ".png") -> str:
    return f"{base}_carrusel_{i}de{n}{ext}"


def borrar_partes(carpeta: Path, base: str) -> None:
    """Elimina las partes de pruebas anteriores de la misma foto (sea cual sea su número)."""
    for n in PARTES:
        for i in range(1, n + 1):
            for ext in (".png", ".jpg"):
                (carpeta / nombre_parte(base, i, n, ext)).unlink(missing_ok=True)


def partes_recomendadas(ancho: float, alto: float, margen: float = 0.12) -> int:
    """Número de partes cuya proporción N:1 deja menos espacio vacío alrededor de un
    sujeto de ancho×alto (con el margen indicado)."""
    m = margen * max(ancho, alto)
    w, h = ancho + 2 * m, alto + 2 * m

    def ocupacion(n: int) -> float:
        lado = max(h, w / n)          # alto del lienzo N:1 mínimo que contiene al sujeto
        return (w * h) / (n * lado * lado)

    return max(PARTES, key=ocupacion)


def partes_de_imagen(imagen: Image.Image) -> int:
    """N de una imagen ya compuesta en modo carrusel (ancho = N × alto), o 0."""
    w, h = imagen.size
    n = w // h if h else 0
    return n if n in PARTES and w == n * h else 0
