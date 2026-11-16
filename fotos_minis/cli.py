"""Uso por línea de comandos (procesa una foto o una carpeta entera)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import perfiles
from .composicion import PROPORCIONES
from .proceso import guardar, procesar
from .segmentacion import MODELOS

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"}

# Opción de la línea de comandos -> campo del perfil. Todas tienen default=None:
# lo que no se indique se toma del perfil (y éste, de los valores de fábrica).
OPCIONES = {
    "modelo": "modelo", "rellenar_huecos": "rellenar_huecos", "halo": "halo",
    "vineteado": "vineteado", "sombra": "sombra", "reflejo": "reflejo",
    "horizonte": "horizonte", "horizonte_distancia": "horizonte_distancia",
    "horizonte_desenfoque": "horizonte_desenfoque", "horizonte_suelo": "horizonte_suelo",
    "horizonte_altura": "horizonte_altura", "oscurecer_borde": "oscurecer_borde",
    "corregir_dominante": "corregir_dominante", "negro": "negro", "margen": "margen",
    "proporcion": "proporcion",
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m fotos_minis",
        description="Sustituye el fondo por un negro de estudio conservando la figura y su peana. "
                    "Por defecto usa el perfil marcado en la aplicación como «Usar al abrir»; "
                    "cualquier opción indicada aquí prevalece sobre el perfil.")
    p.add_argument("entrada", nargs="*", type=Path, help="fotos o carpetas")
    p.add_argument("-o", "--salida", type=Path, default=Path("salida"), help="carpeta de salida")
    p.add_argument("--perfil", help=f"perfil guardado a usar («{perfiles.FABRICA}» para ninguno)")
    p.add_argument("--perfiles", action="store_true", help="muestra los perfiles guardados y sale")
    p.add_argument("--modelo", choices=list(MODELOS))
    p.add_argument("--halo", type=float, help="foco suave detrás (0-1)")
    p.add_argument("--vineteado", type=float)
    p.add_argument("--sombra", type=float, help="sombra de contacto (0-1)")
    p.add_argument("--reflejo", type=float, help="reflejo en suelo brillante (0-1)")
    p.add_argument("--horizonte", type=float, help="intensidad del paso suelo-pared (0 = sin horizonte)")
    p.add_argument("--horizonte-distancia", type=float, help="distancia de la pared, en diámetros de peana")
    p.add_argument("--horizonte-desenfoque", type=float)
    p.add_argument("--horizonte-suelo", type=float, help="luminosidad del suelo (0-1)")
    p.add_argument("--horizonte-altura", type=float,
                   help="ajuste fino de la altura (fracción de la altura de la figura)")
    p.add_argument("--oscurecer-borde", type=float)
    p.add_argument("--corregir-dominante", type=float,
                   help="neutraliza la dominante de color de la luz original (0-1)")
    p.add_argument("--negro", type=int, help="nivel del negro de fondo (0-40)")
    p.add_argument("--margen", type=float)
    p.add_argument("--proporcion", choices=["original", *PROPORCIONES])
    p.add_argument("--sin-reencuadre", action="store_true", help="mantiene el tamaño original")
    p.add_argument("--rellenar-huecos", type=int, metavar="PX",
                   help="rellena huecos interiores de la máscara menores de PX píxeles")
    p.add_argument("--recorte", action="store_true", help="guarda también el PNG transparente")
    p.add_argument("--calidad", type=int, default=95, help="calidad JPEG")
    a = p.parse_args(argv)

    if a.perfiles:
        print(f"Perfiles en {perfiles.ruta()}:")
        por_defecto = perfiles.predeterminado()
        for n in [perfiles.FABRICA, *perfiles.listar()]:
            print(f"  {'*' if n == por_defecto else ' '} {n}")
        print("(* = el que se usa por defecto)")
        return
    if not a.entrada:
        p.error("indica al menos una foto o carpeta")

    nombre = a.perfil or perfiles.predeterminado()
    try:
        valores = perfiles.obtener(nombre)
    except KeyError as e:
        sys.exit(f"{e.args[0]}. Usa --perfiles para ver los disponibles.")
    for opcion, campo in OPCIONES.items():
        if getattr(a, opcion) is not None:
            valores[campo] = getattr(a, opcion)
    if a.sin_reencuadre:
        valores["reencuadrar"] = False
    op, modelo, huecos = perfiles.a_opciones(valores)
    print(f"Perfil: {nombre}")

    ficheros = []
    for e in a.entrada:
        ficheros += sorted(f for f in e.rglob("*") if f.suffix.lower() in EXTS) if e.is_dir() else [e]
    a.salida.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(ficheros, 1):
        print(f"[{i}/{len(ficheros)}] {f}", flush=True)
        r = procesar(f, op, modelo=modelo, rellenar_huecos_px=huecos)
        destino = a.salida / f"{f.stem}_estudio.jpg"
        guardar(r.estudio, destino, quality=a.calidad, subsampling=0)
        if a.recorte:
            guardar(r.recorte, a.salida / f"{f.stem}_recorte.png")
        print(f"    -> {destino}")
