"""Flujo completo: imagen de entrada -> imagen con fondo negro de estudio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from .composicion import OpcionesEstudio, componer, recorte_transparente
from .segmentacion import MODELO_POR_DEFECTO, mascara_bruta, refinar_mascara


@dataclass
class Resultado:
    estudio: Image.Image      # imagen final con fondo negro
    recorte: Image.Image      # sujeto recortado con transparencia (RGBA)
    mascara: Image.Image      # máscara alfa (L)


def cargar(ruta_o_imagen) -> Image.Image:
    img = Image.open(ruta_o_imagen) if not isinstance(ruta_o_imagen, Image.Image) else ruta_o_imagen
    img = ImageOps.exif_transpose(img)  # respeta la orientación de la cámara
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fondo, img)
    return img.convert("RGB")


def procesar(
    imagen,
    opciones: OpcionesEstudio | None = None,
    *,
    modelo: str = MODELO_POR_DEFECTO,
    rellenar_huecos_px: int = 0,
    alfa: np.ndarray | None = None,
) -> Resultado:
    op = opciones or OpcionesEstudio()
    img = cargar(imagen)
    arr = np.asarray(img)
    if alfa is None:
        alfa = mascara_bruta(img, modelo)
        alfa = refinar_mascara(arr, alfa, rellenar_huecos_px=rellenar_huecos_px)
    salida = componer(arr, alfa, op)
    rgba = recorte_transparente(arr, alfa, op.descontaminar_bordes)
    return Resultado(
        estudio=Image.fromarray(salida),
        recorte=Image.fromarray(rgba, "RGBA"),
        mascara=Image.fromarray(np.round(alfa * 255).astype(np.uint8), "L"),
    )
