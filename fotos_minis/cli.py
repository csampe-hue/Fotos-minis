"""Uso por línea de comandos (procesa una foto o una carpeta entera)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .composicion import PROPORCIONES, OpcionesEstudio
from .proceso import guardar, procesar
from .segmentacion import MODELO_POR_DEFECTO, MODELOS

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m fotos_minis",
        description="Sustituye el fondo por un negro de estudio conservando la figura y su peana.")
    p.add_argument("entrada", nargs="+", type=Path, help="fotos o carpetas")
    p.add_argument("-o", "--salida", type=Path, default=Path("salida"), help="carpeta de salida")
    p.add_argument("--modelo", default=MODELO_POR_DEFECTO, choices=list(MODELOS))
    p.add_argument("--halo", type=float, default=0.10, help="foco suave detrás (0-1)")
    p.add_argument("--vineteado", type=float, default=0.35)
    p.add_argument("--sombra", type=float, default=0.6, help="sombra de contacto (0-1)")
    p.add_argument("--reflejo", type=float, default=0.0, help="reflejo en suelo brillante (0-1)")
    p.add_argument("--oscurecer-borde", type=float, default=0.15)
    p.add_argument("--corregir-dominante", type=float, default=0.0,
                   help="neutraliza la dominante de color de la luz original (0-1)")
    p.add_argument("--negro", type=int, default=0, help="nivel del negro de fondo (0-40)")
    p.add_argument("--margen", type=float, default=0.12)
    p.add_argument("--proporcion", default="original", choices=["original", *PROPORCIONES])
    p.add_argument("--sin-reencuadre", action="store_true", help="mantiene el tamaño original")
    p.add_argument("--rellenar-huecos", type=int, default=0, metavar="PX",
                   help="rellena huecos interiores de la máscara menores de PX píxeles")
    p.add_argument("--recorte", action="store_true", help="guarda también el PNG transparente")
    p.add_argument("--calidad", type=int, default=95, help="calidad JPEG")
    a = p.parse_args(argv)

    op = OpcionesEstudio(negro=a.negro, halo=a.halo, vineteado=a.vineteado, sombra=a.sombra,
                         reflejo=a.reflejo, oscurecer_borde=a.oscurecer_borde,
                         corregir_dominante=a.corregir_dominante, margen=a.margen,
                         proporcion=a.proporcion, reencuadrar=not a.sin_reencuadre)
    ficheros = []
    for e in a.entrada:
        ficheros += sorted(f for f in e.rglob("*") if f.suffix.lower() in EXTS) if e.is_dir() else [e]
    a.salida.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(ficheros, 1):
        print(f"[{i}/{len(ficheros)}] {f}", flush=True)
        r = procesar(f, op, modelo=a.modelo, rellenar_huecos_px=a.rellenar_huecos)
        destino = a.salida / f"{f.stem}_estudio.jpg"
        guardar(r.estudio, destino, quality=a.calidad, subsampling=0)
        if a.recorte:
            guardar(r.recorte, a.salida / f"{f.stem}_recorte.png")
        print(f"    -> {destino}")
