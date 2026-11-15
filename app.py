"""Interfaz web: python app.py  ->  abre http://127.0.0.1:7860"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from fotos_minis.composicion import PROPORCIONES, OpcionesEstudio, _pegar, calcular_lienzo, componer
from fotos_minis.proceso import cargar, procesar
from fotos_minis.segmentacion import MODELO_POR_DEFECTO, MODELOS, mascara_bruta, refinar_mascara

_cache_mascaras: dict[str, np.ndarray] = {}


def _clave(arr: np.ndarray, modelo: str, huecos: int) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest() + f"|{modelo}|{huecos}"


def _mascara(arr: np.ndarray, modelo: str, huecos: int) -> np.ndarray:
    k = _clave(arr, modelo, huecos)
    if k not in _cache_mascaras:
        if len(_cache_mascaras) > 8:
            _cache_mascaras.clear()
        alfa = mascara_bruta(Image.fromarray(arr), modelo)
        _cache_mascaras[k] = refinar_mascara(arr, alfa, rellenar_huecos_px=int(huecos))
    return _cache_mascaras[k]


def _opciones(halo, vineteado, sombra, reflejo, caida, oscurecer, dominante, negro, reencuadrar,
              margen, proporcion, descontaminar) -> OpcionesEstudio:
    return OpcionesEstudio(halo=halo, vineteado=vineteado, sombra=sombra, reflejo=reflejo,
                           reflejo_caida=caida, oscurecer_borde=oscurecer,
                           corregir_dominante=dominante, negro=int(negro), reencuadrar=reencuadrar,
                           margen=margen, proporcion=proporcion, descontaminar_bordes=descontaminar)


def procesar_una(ruta, modelo, huecos, *ajustes):
    if ruta is None:
        return None, None, None
    img = cargar(ruta)
    arr = np.asarray(img)
    alfa = _mascara(arr, modelo, huecos)
    op = _opciones(*ajustes)
    out = componer(arr, alfa, op)
    # "Antes" con el mismo encuadre que el resultado, para que el comparador coincida.
    ox, oy, W, H = calcular_lienzo(alfa, op)
    antes = np.zeros((H, W, 3), np.uint8)
    _pegar(antes, arr, ox, oy)
    destino = Path(tempfile.mkdtemp()) / f"{Path(ruta).stem}_estudio.png"
    Image.fromarray(out).save(destino)
    mascara = Image.fromarray(np.round(alfa * 255).astype(np.uint8))
    return (Image.fromarray(antes), Image.fromarray(out)), mascara, str(destino)


def procesar_lote(ficheros, modelo, huecos, *ajustes, progress=gr.Progress()):
    if not ficheros:
        return None
    op = _opciones(*ajustes)
    carpeta = Path(tempfile.mkdtemp())
    zip_path = carpeta / "fotos_estudio.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in progress.tqdm(ficheros, desc="Procesando"):
            r = procesar(f, op, modelo=modelo, rellenar_huecos_px=int(huecos))
            nombre = f"{Path(f).stem}_estudio.png"
            r.estudio.save(carpeta / nombre)
            z.write(carpeta / nombre, nombre)
    return str(zip_path)


with gr.Blocks(title="Fotos Minis · Fondo negro de estudio") as demo:
    gr.Markdown(
        "# Fotos Minis · Fondo negro de estudio\n"
        "Sube la foto de tu figura: se sustituye el fondo por un negro de estudio "
        "conservando intactos la figura y la peana. Ajusta el acabado con los controles.")
    with gr.Row():
        with gr.Column(scale=1):
            entrada = gr.Image(label="Foto original", type="filepath", sources=["upload", "clipboard"])
            with gr.Accordion("Recorte", open=True):
                modelo = gr.Dropdown([(v, k) for k, v in MODELOS.items()], value=MODELO_POR_DEFECTO,
                                     label="Modelo de segmentación")
                huecos = gr.Slider(0, 5000, 0, step=50, label="Rellenar huecos pequeños de la máscara (px)",
                                   info="Úsalo si aparecen agujeros en zonas oscuras de la figura")
                descontaminar = gr.Checkbox(True, label="Limpiar el color del fondo antiguo en el contorno")
            with gr.Accordion("Luz y sombras", open=True):
                halo = gr.Slider(0, 1, 0.10, step=0.01, label="Foco suave detrás de la figura")
                vineteado = gr.Slider(0, 1, 0.35, step=0.01, label="Viñeteado")
                sombra = gr.Slider(0, 1, 0.6, step=0.01, label="Sombra de contacto bajo la peana")
                reflejo = gr.Slider(0, 1, 0.0, step=0.01, label="Reflejo en suelo negro brillante")
                caida = gr.Slider(0.05, 0.6, 0.25, step=0.01, label="Longitud del reflejo")
                oscurecer = gr.Slider(0, 0.6, 0.15, step=0.01, label="Integración del borde con el negro")
                dominante = gr.Slider(0, 1, 0.0, step=0.01, label="Corregir dominante de color de la luz",
                                      info="0 = colores originales intactos")
                negro = gr.Slider(0, 40, 0, step=1, label="Nivel del negro de fondo")
            with gr.Accordion("Encuadre", open=False):
                reencuadrar = gr.Checkbox(True, label="Reencuadrar centrando la figura (sin reescalarla)")
                margen = gr.Slider(0, 0.5, 0.12, step=0.01, label="Margen")
                proporcion = gr.Dropdown(["original", *PROPORCIONES], value="original", label="Proporción")
            boton = gr.Button("Procesar", variant="primary")
        with gr.Column(scale=2):
            comparador = gr.ImageSlider(label="Antes / después", type="pil")
            with gr.Row():
                descarga = gr.File(label="Descargar (PNG sin pérdida)")
                mascara = gr.Image(label="Máscara", type="pil", height=200)

    with gr.Accordion("Procesar varias fotos a la vez", open=False):
        lote = gr.File(file_count="multiple", file_types=["image"], label="Fotos")
        boton_lote = gr.Button("Procesar lote con los ajustes actuales")
        zip_salida = gr.File(label="ZIP con los resultados")

    ajustes = [halo, vineteado, sombra, reflejo, caida, oscurecer, dominante, negro, reencuadrar,
               margen, proporcion, descontaminar]
    entradas = [entrada, modelo, huecos, *ajustes]
    salidas = [comparador, mascara, descarga]
    boton.click(procesar_una, entradas, salidas)
    entrada.change(procesar_una, entradas, salidas)
    for c in [*ajustes, modelo, huecos]:
        c.release(procesar_una, entradas, salidas) if isinstance(c, gr.Slider) \
            else c.change(procesar_una, entradas, salidas)
    boton_lote.click(procesar_lote, [lote, modelo, huecos, *ajustes], zip_salida)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
