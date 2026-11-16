"""Interfaz web: python app.py  ->  abre http://127.0.0.1:7860"""

from __future__ import annotations

import hashlib
import os
import signal
import threading
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from fotos_minis.composicion import PROPORCIONES, OpcionesEstudio, _pegar, calcular_lienzo, componer
from fotos_minis.proceso import cargar, guardar, procesar
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
              margen, proporcion, descontaminar, horizonte, h_distancia, h_desenfoque,
              h_suelo, h_altura) -> OpcionesEstudio:
    return OpcionesEstudio(halo=halo, vineteado=vineteado, sombra=sombra, reflejo=reflejo,
                           reflejo_caida=caida, oscurecer_borde=oscurecer,
                           corregir_dominante=dominante, negro=int(negro), reencuadrar=reencuadrar,
                           margen=margen, proporcion=proporcion, descontaminar_bordes=descontaminar,
                           horizonte=horizonte, horizonte_distancia=h_distancia,
                           horizonte_desenfoque=h_desenfoque, horizonte_suelo=h_suelo,
                           horizonte_altura=h_altura)


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
    resultado = Image.fromarray(out)
    resultado.info.update(img.info)
    guardar(resultado, destino)
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
            guardar(r.estudio, carpeta / nombre)
            z.write(carpeta / nombre, nombre)
    return str(zip_path)


def cerrar_aplicacion():
    """Detiene el servidor de forma ordenada: equivale a pulsar Ctrl + C en la terminal
    (Gradio cierra el servidor y el programa termina). Se hace con un pequeño retraso
    para que el navegador reciba antes la respuesta."""
    threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    return gr.update(value="Cerrando…", interactive=False)


AVISO_CIERRE = """() => {
    setTimeout(() => {
        document.body.innerHTML = '<div style="font-family:sans-serif;max-width:32rem;margin:20vh auto;'
            + 'text-align:center;color:#333"><h2>Fotos Minis se ha cerrado</h2>'
            + '<p>Ya puedes cerrar esta pestaña. Para volver a usarla, abre de nuevo el lanzador.</p></div>';
    }, 1500);
}"""


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
            with gr.Accordion("Horizonte (paso del suelo a la pared)", open=False):
                gr.Markdown("La línea se sitúa según la perspectiva de la peana y se desenfoca "
                            "más cuanto más lejos queda la pared.")
                horizonte = gr.Slider(0, 1, 0.0, step=0.01, label="Intensidad del horizonte",
                                      info="0 = fondo negro continuo, sin horizonte")
                h_distancia = gr.Slider(0.5, 10, 3.0, step=0.1, label="Distancia de la pared",
                                        info="En diámetros de peana detrás de la figura")
                h_desenfoque = gr.Slider(0, 3, 0.5, step=0.05, label="Desenfoque de la línea")
                h_suelo = gr.Slider(0, 1, 0.5, step=0.01, label="Luminosidad del suelo")
                h_altura = gr.Slider(-0.5, 0.5, 0.0, step=0.01, label="Ajuste fino de la altura",
                                     info="Corrige la posición si la perspectiva estimada no convence")
            with gr.Accordion("Encuadre", open=False):
                reencuadrar = gr.Checkbox(True, label="Reencuadrar centrando la figura en horizontal y vertical (sin reescalarla)")
                margen = gr.Slider(0, 0.5, 0.12, step=0.01, label="Margen")
                proporcion = gr.Dropdown(["original", *PROPORCIONES], value="original", label="Proporción")
            boton = gr.Button("Procesar", variant="primary")
        with gr.Column(scale=2):
            comparador = gr.ImageSlider(label="Antes / después", type="pil")
            with gr.Row():
                descarga = gr.File(label="Descargar (PNG sin pérdida)")
                mascara = gr.Image(label="Máscara", type="pil", height=200)

    with gr.Accordion("Procesar varias fotos a la vez", open=False):
        lote = gr.File(file_count="multiple", file_types=["image", ".heic", ".heif"], label="Fotos")
        boton_lote = gr.Button("Procesar lote con los ajustes actuales")
        zip_salida = gr.File(label="ZIP con los resultados")

    with gr.Row():
        gr.Markdown("Al terminar, cierra la aplicación con este botón (libera la memoria de los modelos).")
        boton_cerrar = gr.Button("Cerrar aplicación", variant="stop", scale=0, min_width=200)

    ajustes = [halo, vineteado, sombra, reflejo, caida, oscurecer, dominante, negro, reencuadrar,
               margen, proporcion, descontaminar, horizonte, h_distancia, h_desenfoque, h_suelo, h_altura]
    entradas = [entrada, modelo, huecos, *ajustes]
    salidas = [comparador, mascara, descarga]
    boton.click(procesar_una, entradas, salidas)
    entrada.change(procesar_una, entradas, salidas)
    for c in [*ajustes, modelo, huecos]:
        c.release(procesar_una, entradas, salidas) if isinstance(c, gr.Slider) \
            else c.change(procesar_una, entradas, salidas)
    boton_cerrar.click(cerrar_aplicacion, None, boton_cerrar, js=AVISO_CIERRE, queue=False)
    boton_lote.click(procesar_lote, [lote, modelo, huecos, *ajustes], zip_salida)


if __name__ == "__main__":
    # Si se arranca en segundo plano (p. ej. desde el lanzador), el shell hace que
    # se ignore Ctrl + C; se restaura para que el botón "Cerrar aplicación" funcione.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    demo.queue(default_concurrency_limit=1).launch()
