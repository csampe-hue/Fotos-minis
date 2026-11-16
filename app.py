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

from fotos_minis.composicion import (PROPORCIONES, OpcionesEstudio, _pegar, calcular_lienzo, componer,
                                     descontaminar as limpiar_primer_plano)
from fotos_minis.proceso import cargar, guardar, procesar
from fotos_minis.segmentacion import MODELO_POR_DEFECTO, MODELOS, mascara_bruta, refinar_mascara

_cache_mascaras: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _clave(arr: np.ndarray, modelo: str, huecos: int) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest() + f"|{modelo}|{huecos}"


def _mascara(arr: np.ndarray, modelo: str, huecos: int) -> tuple[np.ndarray, np.ndarray]:
    k = _clave(arr, modelo, huecos)
    if k not in _cache_mascaras:
        if len(_cache_mascaras) > 8:
            _cache_mascaras.clear()
        alfa = mascara_bruta(Image.fromarray(arr), modelo)
        alfa = refinar_mascara(arr, alfa, rellenar_huecos_px=int(huecos))
        # El primer plano limpio (lento) se calcula una vez por máscara: así mover los
        # controles de luz, horizonte o encuadre sólo recompone el fondo.
        _cache_mascaras[k] = (alfa, limpiar_primer_plano(arr, alfa))
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
    alfa, primer_plano = _mascara(arr, modelo, huecos)
    op = _opciones(*ajustes)
    out = componer(arr, alfa, op, primer_plano)
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


CSS = """
.gradio-container { max-width: 100% !important; padding: 10px 20px 16px !important; }
#cabecera { align-items: center; gap: 12px; }
#cabecera h1 { margin: 0 0 2px; font-size: 1.45rem; }
#cabecera p { margin: 0; opacity: .75; }
#boton-cerrar { align-self: center; }
/* Panel de ajustes: ocupa la altura de la ventana y se desplaza por su cuenta. */
#panel { position: sticky; top: 8px; max-height: calc(100vh - 150px); overflow-y: auto;
         overflow-x: hidden; padding-right: 4px; flex-wrap: nowrap !important; }
#panel > * { flex-shrink: 0; }
#visor-tabs .tabitem { padding: 0 !important; }
#comparador, #mascara { height: calc(100vh - 260px) !important; min-height: 360px; }
/* Las imágenes ocupan todo el alto disponible del visor, sin deformarse. */
#comparador .content, #comparador .content img,
#mascara .image-container, #mascara img { height: 100% !important; max-height: none !important; }
#comparador img, #mascara img { width: 100% !important; object-fit: contain; }
@media (max-width: 960px) {
  #panel { position: static; max-height: none; overflow: visible; }
  #comparador, #mascara { height: 70vh !important; }
}
"""

with gr.Blocks(title="Fotos Minis · Fondo negro de estudio", fill_width=True) as demo:
    with gr.Row(elem_id="cabecera", equal_height=True):
        gr.Markdown(
            "# Fotos Minis · Fondo negro de estudio\n"
            "Sustituye el fondo por un negro de estudio conservando intactos la figura y la peana.")
        boton_cerrar = gr.Button("Cerrar aplicación", variant="stop", scale=0, min_width=180,
                                 elem_id="boton-cerrar")

    with gr.Tabs():
        with gr.Tab("Una foto", id="una"):
            with gr.Row(equal_height=False):
                # ---------------- Panel de ajustes ----------------
                with gr.Column(scale=1, min_width=330, elem_id="panel"):
                    entrada = gr.Image(label="Foto original", type="filepath", height=210,
                                       sources=["upload", "clipboard"])
                    boton = gr.Button("Volver a procesar", variant="primary", size="sm")
                    with gr.Tabs():
                        with gr.Tab("Recorte", id="recorte"):
                            modelo = gr.Dropdown([(v, k) for k, v in MODELOS.items()],
                                                 value=MODELO_POR_DEFECTO, label="Modelo de segmentación")
                            huecos = gr.Slider(0, 5000, 0, step=50, label="Rellenar huecos de la máscara (px)",
                                               info="Si aparecen agujeros en zonas oscuras de la figura")
                            descontaminar = gr.Checkbox(True, label="Limpiar el color del fondo antiguo en el contorno")
                        with gr.Tab("Luz", id="luz"):
                            halo = gr.Slider(0, 1, 0.10, step=0.01, label="Foco suave detrás de la figura")
                            vineteado = gr.Slider(0, 1, 0.35, step=0.01, label="Viñeteado")
                            sombra = gr.Slider(0, 1, 0.6, step=0.01, label="Sombra de contacto bajo la peana")
                            reflejo = gr.Slider(0, 1, 0.0, step=0.01, label="Reflejo en suelo negro brillante")
                            caida = gr.Slider(0.05, 0.6, 0.25, step=0.01, label="Longitud del reflejo")
                            oscurecer = gr.Slider(0, 0.6, 0.15, step=0.01, label="Integración del borde con el negro")
                            dominante = gr.Slider(0, 1, 0.0, step=0.01, label="Corregir dominante de color",
                                                  info="0 = colores originales intactos")
                            negro = gr.Slider(0, 40, 0, step=1, label="Nivel del negro de fondo")
                        with gr.Tab("Horizonte", id="horizonte"):
                            gr.Markdown("Paso del suelo a la pared. Se sitúa según la perspectiva de la "
                                        "peana y se desenfoca más cuanto más lejos queda la pared.")
                            horizonte = gr.Slider(0, 1, 0.0, step=0.01, label="Intensidad",
                                                  info="0 = fondo negro continuo, sin horizonte")
                            h_distancia = gr.Slider(0.5, 10, 3.0, step=0.1, label="Distancia de la pared",
                                                    info="En diámetros de peana detrás de la figura")
                            h_desenfoque = gr.Slider(0, 3, 0.5, step=0.05, label="Desenfoque de la línea")
                            h_suelo = gr.Slider(0, 1, 0.5, step=0.01, label="Luminosidad del suelo")
                            h_altura = gr.Slider(-0.5, 0.5, 0.0, step=0.01, label="Ajuste fino de la altura",
                                                 info="Por si la perspectiva estimada no convence")
                        with gr.Tab("Encuadre", id="encuadre"):
                            reencuadrar = gr.Checkbox(True, label="Centrar la figura (sin reescalarla)")
                            margen = gr.Slider(0, 0.5, 0.12, step=0.01, label="Margen")
                            proporcion = gr.Dropdown(["original", *PROPORCIONES], value="original",
                                                     label="Proporción")

                # ---------------- Visor ----------------
                with gr.Column(scale=3, min_width=480):
                    with gr.Tabs(elem_id="visor-tabs"):
                        with gr.Tab("Antes / después", id="comparar"):
                            comparador = gr.ImageSlider(show_label=False, type="pil", elem_id="comparador")
                        with gr.Tab("Máscara", id="mascara_tab"):
                            mascara = gr.Image(show_label=False, type="pil", elem_id="mascara")
                    descarga = gr.DownloadButton("Descargar resultado (PNG sin pérdida)", variant="primary")

        with gr.Tab("Varias fotos", id="varias"):
            gr.Markdown("Se procesan con los ajustes elegidos en la pestaña **Una foto**.")
            with gr.Row(equal_height=True):
                lote = gr.File(file_count="multiple", file_types=["image", ".heic", ".heif"],
                               label="Fotos", scale=2)
                with gr.Column(scale=1, min_width=260):
                    boton_lote = gr.Button("Procesar todas", variant="primary")
                    zip_salida = gr.File(label="ZIP con los resultados")

    ajustes = [halo, vineteado, sombra, reflejo, caida, oscurecer, dominante, negro, reencuadrar,
               margen, proporcion, descontaminar, horizonte, h_distancia, h_desenfoque, h_suelo, h_altura]
    entradas = [entrada, modelo, huecos, *ajustes]
    salidas = [comparador, mascara, descarga]
    boton.click(procesar_una, entradas, salidas)
    entrada.change(procesar_una, entradas, salidas)
    # "always_last": al arrastrar un control sólo se procesa el último valor, y
    # también reacciona cuando se escribe el número en la casilla.
    for c in [*ajustes, modelo, huecos]:
        c.change(procesar_una, entradas, salidas, trigger_mode="always_last", show_progress="minimal")
    boton_cerrar.click(cerrar_aplicacion, None, boton_cerrar, js=AVISO_CIERRE, queue=False)
    boton_lote.click(procesar_lote, [lote, modelo, huecos, *ajustes], zip_salida)


if __name__ == "__main__":
    # Si se arranca en segundo plano (p. ej. desde el lanzador), el shell hace que
    # se ignore Ctrl + C; se restaura para que el botón "Cerrar aplicación" funcione.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
