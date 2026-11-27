"""Interfaz web: python app.py  ->  abre http://127.0.0.1:7860"""

from __future__ import annotations

import hashlib
from dataclasses import replace
import os
import shutil
from datetime import datetime
import signal
import threading
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from fotos_minis.composicion import (PROPORCIONES, SUELOS, OpcionesEstudio, _pegar, calcular_lienzo, componer,
                                     mascara_suelo,
                                     descontaminar as limpiar_primer_plano)
from fotos_minis import perfiles, preferencias
from fotos_minis.perfiles import CAMPOS_ESTUDIO, FABRICA
from fotos_minis.carrusel import borrar_partes, dividir, nombre_parte, partes_de_imagen
from fotos_minis.composicion import CARRUSEL_AUTO, partes_carrusel
from fotos_minis.individuales import borrar_individuales, individuales as fotos_individuales, nombre_individual
from fotos_minis.composicion import caja_sujeto
from fotos_minis.proceso import cargar, guardar, procesar
from fotos_minis.superres import ampliar_grupo
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


def _opciones(*ajustes) -> OpcionesEstudio:
    """Ajustes de la interfaz (en el orden de CAMPOS_ESTUDIO) -> OpcionesEstudio."""
    return perfiles.a_opciones(dict(zip(CAMPOS_ESTUDIO, ajustes)))[0]


# Última combinación (foto + ajustes) procesada en cada pestaña del navegador.
_ultimas_firmas: dict[str, tuple] = {}


def _sesion(request: gr.Request | None) -> str:
    return getattr(request, "session_hash", None) or "local"


def carpeta_fotos() -> Path:
    """Carpeta de guardado elegida por el usuario (la última utilizada)."""
    return preferencias.carpeta_guardado()


def _guardar_copia(origen: Path, nombre: str) -> Path:
    """Copia el resultado a la carpeta «fotos». Con el mismo nombre se sobrescribe:
    al hacer varias pruebas con la misma foto original sólo queda la última."""
    destino = carpeta_fotos() / nombre
    tmp = destino.with_name(destino.name + ".tmp")
    shutil.copyfile(origen, tmp)
    tmp.replace(destino)  # sustitución atómica: nunca queda un fichero a medias
    return destino


def _aviso_guardado(ruta: Path) -> str:
    try:
        mostrar = ruta.relative_to(Path.home())
        mostrar = f"~/{mostrar}"
    except ValueError:
        mostrar = ruta
    return f"Guardado automáticamente en `{mostrar}`"


def cambiar_carpeta(texto):
    actual = carpeta_fotos()
    if texto and Path(str(texto).strip()).expanduser() == actual:
        return gr.skip()  # sin cambios (p. ej. al salir del campo tras pulsar Intro)
    try:
        carpeta = preferencias.fijar_carpeta_guardado(texto)
    except ValueError as e:
        gr.Warning(str(e))
        return str(carpeta_fotos())
    gr.Info(f"Las fotos se guardarán en {carpeta}")
    return str(carpeta)


def elegir_carpeta(actual):
    elegida = preferencias.elegir_carpeta_sistema(Path(actual) if actual else carpeta_fotos())
    if elegida is None:
        return gr.skip()
    return cambiar_carpeta(elegida)


def abrir_carpeta():
    carpeta = carpeta_fotos()
    if not preferencias.abrir_en_explorador(carpeta):
        gr.Info(f"Carpeta: {carpeta}")


def procesar_una(ruta, modelo, huecos, *ajustes, request: gr.Request = None, progress=gr.Progress()):
    sesion = _sesion(request)
    if ruta is None:
        # Al sustituir una foto por otra el componente pasa un instante por "vacío":
        # no se borra el resultado (sólo se borra al pulsar la X, ver limpiar()).
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    # Al aplicar un perfil cambian muchos controles a la vez y cada uno lanza un
    # evento: si los valores son los mismos que los del último resultado, no se repite.
    firma = (ruta, modelo, huecos, *ajustes)
    if firma == _ultimas_firmas.get(sesion):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    img = cargar(ruta)
    arr = np.asarray(img)
    if _clave(arr, modelo, huecos) not in _cache_mascaras:
        progress(0.1, desc="Recortando la figura… (la primera foto tarda más: se cargan los modelos)")
    alfa, primer_plano = _mascara(arr, modelo, huecos)
    progress(0.9, desc="Componiendo el fondo…")
    op = _opciones(*ajustes)
    suelo_m = None
    if op.suelo == "original":
        suelo_m = mascara_suelo(arr, alfa, op)
        if suelo_m is None:
            gr.Warning("No se distingue el suelo de la pared en esta foto: se usa el suelo de estudio.")
    if op.nitidez > 0:
        progress(0.92, desc="Corrigiendo el desenfoque por distancia…")
    out = componer(arr, alfa, op, primer_plano, suelo_mascara=suelo_m)
    # "Antes" con el mismo encuadre que el resultado, para que el comparador coincida.
    ox, oy, W, H = calcular_lienzo(alfa, op, dentro_de_la_foto=suelo_m is not None)
    antes = np.zeros((H, W, 3), np.uint8)
    _pegar(antes, arr, ox, oy)
    # Foto de grupo («madre»): super-resolución opcional. Las individuales se sacan de
    # ``out`` sin ampliar (ellas se amplían por su cuenta).
    grupo = out
    if op.superres in (2, 4) and op.superres_grupo:
        progress(0.95, desc="Super-resolución de la foto de grupo…")
        grupo, aviso_sr = ampliar_grupo(out, op.superres, partes_carrusel(alfa, op))
        if aviso_sr:
            gr.Warning(aviso_sr)
        if grupo.shape != out.shape:
            antes = np.asarray(Image.fromarray(antes).resize(grupo.shape[1::-1], Image.BICUBIC))
    base = Path(ruta).stem
    temporal = Path(tempfile.mkdtemp())
    destino = temporal / f"{base}_estudio.png"
    resultado = Image.fromarray(grupo)
    resultado.info.update(img.info)
    guardar(resultado, destino)
    guardado = _guardar_copia(destino, destino.name)
    aviso = _aviso_guardado(guardado)
    descarga = destino

    # Carrusel: además de la imagen completa (sin cortar) se guardan las N partes
    # cuadradas. Las partes de pruebas anteriores de esta foto se eliminan siempre.
    borrar_partes(carpeta_fotos(), base)
    galeria = None
    n = partes_carrusel(alfa, op)
    if n:
        a_l = np.zeros((H, W), np.float32)
        _pegar(a_l, alfa, ox, oy)
        x0, y0, x1, y1 = caja_sujeto(a_l, umbral=0.5)
        if (alfa > 0.5).sum() > (a_l > 0.5).sum() * 1.01:
            gr.Warning("La foto no es lo bastante ancha para este carrusel sin salirse de ella: "
                       "parte de las figuras queda fuera. Prueba con menos partes o con suelo de estudio.")
        galeria = []
        descarga = temporal / f"{base}_carrusel.zip"
        with zipfile.ZipFile(descarga, "w") as z:
            z.write(destino, destino.name)
            for i, parte in enumerate(dividir(resultado, n), 1):
                nombre = nombre_parte(base, i, n)
                guardar(parte, temporal / nombre)
                _guardar_copia(temporal / nombre, nombre)
                z.write(temporal / nombre, nombre)
                galeria.append((str(temporal / nombre), f"{i} de {n}"))
        aviso += f" y las {n} partes del carrusel (`{nombre_parte(base, 1, n)}`…)"
        if op.carrusel == CARRUSEL_AUTO:
            aviso += f". Automático: **{n} partes** es lo que mejor aprovecha esta foto."
        else:
            mejor = partes_carrusel(alfa, replace(op, carrusel=CARRUSEL_AUTO))
            if mejor != n:
                aviso += (f". Para esta foto, con **{mejor} partes** las minis ocuparían más espacio "
                          "(o elige «Auto»).")

    # Fotos individuales: una por figura (si hay varias), centrada y entera; recortada
    # de la foto de grupo o, si no cabe o asoman otras figuras, con fondo nuevo.
    borrar_individuales(carpeta_fotos(), base)
    individuales = None
    if op.individuales:
        avisos_sr: list[str] = []
        lista, tapadas = fotos_individuales(out, alfa, op, arr, primer_plano, avisos=avisos_sr)
        for msg in avisos_sr:
            gr.Warning(msg)
        trozos = [im for im, _ in lista]
        if trozos:
            individuales = []
            if descarga == destino:                       # aún no hay ZIP (sin carrusel)
                descarga = temporal / f"{base}_fotos.zip"
                with zipfile.ZipFile(descarga, "w") as z:
                    z.write(destino, destino.name)
            with zipfile.ZipFile(descarga, "a") as z:
                for i, trozo in enumerate(trozos, 1):
                    nombre = nombre_individual(base, i)
                    im = Image.fromarray(trozo)
                    im.info.update(img.info)
                    guardar(im, temporal / nombre)
                    _guardar_copia(temporal / nombre, nombre)
                    z.write(temporal / nombre, nombre)
                    individuales.append((str(temporal / nombre), f"Figura {i} · {lista[i - 1][1]}"))
            aviso += ("" if aviso.endswith(".") else ".") + f" Además, {len(trozos)} fotos individuales (`{nombre_individual(base, 1)}`…)"
            if tapadas:
                aviso += (f"; {tapadas} figura{'s' if tapadas > 1 else ''} casi oculta{'s' if tapadas > 1 else ''} "
                          "tras otras no se puede{} mostrar entera{} y se omite{}.").format(
                              *(("n", "s", "n") if tapadas > 1 else ("", "", "")))
        else:
            aviso += ("" if aviso.endswith(".") else ".") + " Sólo se detecta una figura: no se generan fotos individuales."

    mascara = Image.fromarray(np.round(alfa * 255).astype(np.uint8))
    _ultimas_firmas[sesion] = firma
    return ((Image.fromarray(antes), resultado), mascara, str(descarga), aviso, galeria,
            individuales)


def procesar_desde_ui(request: gr.Request, progress=gr.Progress(), *entradas):
    """Punto de entrada de la interfaz. Gradio sólo inyecta la sesión y la barra de
    progreso en parámetros posicionales anteriores a *args, por eso van primero."""
    return procesar_una(*entradas, request=request, progress=progress)


# --------------------------------------------------------------------------- #
# Perfiles
# --------------------------------------------------------------------------- #

def _opciones_perfil(seleccion: str | None = None):
    return gr.update(choices=[FABRICA, *perfiles.listar()], value=seleccion)


def _estado_perfiles(mensaje: str = "") -> str:
    texto = f"Al abrir se aplica: **{perfiles.predeterminado()}**"
    return f"{mensaje}\n\n{texto}" if mensaje else texto


def _valores_controles(valores: dict) -> list:
    return [valores[k] for k in perfiles.CAMPOS]


def al_abrir():
    """Al cargar la página se aplican los ajustes del perfil predeterminado."""
    nombre = perfiles.predeterminado()
    return [*_valores_controles(perfiles.obtener(nombre)), _opciones_perfil(nombre),
            _estado_perfiles(), preferencias.apariencia(), str(carpeta_fotos())]


def guardar_apariencia(modo):
    preferencias.fijar_apariencia(modo)


def aplicar_perfil(nombre):
    try:
        valores = perfiles.obtener(nombre)
    except KeyError as e:
        raise gr.Error(str(e))
    return [*_valores_controles(valores), _estado_perfiles(f"Aplicado «{nombre or FABRICA}».")]


def guardar_perfil(nombre, al_abrir_tambien, *controles):
    try:
        nombre = perfiles.guardar(nombre, dict(zip(perfiles.CAMPOS, controles)), al_abrir_tambien)
    except ValueError as e:
        raise gr.Error(str(e))
    return _opciones_perfil(nombre), "", _estado_perfiles(f"Guardado «{nombre}».")


def usar_al_abrir(nombre):
    perfiles.fijar_predeterminado(nombre)
    return _estado_perfiles(f"«{nombre or FABRICA}» se aplicará al abrir la aplicación.")


def borrar_perfil(nombre):
    if not nombre or nombre == FABRICA:
        raise gr.Error("Los valores de fábrica no se pueden borrar.")
    perfiles.borrar(nombre)
    return _opciones_perfil(FABRICA), _estado_perfiles(f"Borrado «{nombre}».")


def _grupo_ampliado(r, op: OpcionesEstudio, nombre: str) -> Image.Image:
    """La foto de grupo del lote, con super-resolución si se ha pedido para ella."""
    if op.superres not in (2, 4) or not op.superres_grupo:
        return r.estudio
    alfa = np.asarray(r.mascara, np.float32) / 255.0
    arr, aviso = ampliar_grupo(np.asarray(r.estudio), op.superres, partes_carrusel(alfa, op))
    if aviso:
        gr.Warning(f"{nombre}: {aviso}")
    im = Image.fromarray(arr)
    im.info.update(r.estudio.info)
    return im


def procesar_lote(ficheros, modelo, huecos, *ajustes, progress=gr.Progress()):
    if not ficheros:
        return None, ""
    op = _opciones(*ajustes)
    carpeta = Path(tempfile.mkdtemp())
    zip_path = carpeta / "fotos_estudio.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in progress.tqdm(ficheros, desc="Procesando"):
            r = procesar(f, op, modelo=modelo, rellenar_huecos_px=int(huecos))
            estudio = _grupo_ampliado(r, op, Path(f).name)
            nombre = f"{Path(f).stem}_estudio.png"
            guardar(estudio, carpeta / nombre)
            z.write(carpeta / nombre, nombre)
            n = partes_de_imagen(estudio) if op.carrusel else 0
            if n:
                for i, parte in enumerate(dividir(estudio, n), 1):
                    nombre = nombre_parte(Path(f).stem, i, n)
                    guardar(parte, carpeta / nombre)
                    z.write(carpeta / nombre, nombre)
            if op.individuales:
                alfa_f = np.asarray(r.mascara, np.float32) / 255.0
                avisos_sr: list[str] = []
                lista, _ = fotos_individuales(np.asarray(r.estudio), alfa_f, op, np.asarray(cargar(f)),
                                              avisos=avisos_sr)
                for msg in avisos_sr:
                    gr.Warning(f"{Path(f).name}: {msg}")
                for i, (trozo, _) in enumerate(lista, 1):
                    nombre = nombre_individual(Path(f).stem, i)
                    guardar(Image.fromarray(trozo), carpeta / nombre)
                    z.write(carpeta / nombre, nombre)
    guardado = _guardar_copia(zip_path, f"fotos_estudio_{datetime.now():%Y-%m-%d_%H-%M-%S}.zip")
    return str(zip_path), _aviso_guardado(guardado)


def limpiar(request: gr.Request = None):
    """La X de la foto original: se vacía también el resultado."""
    _ultimas_firmas.pop(_sesion(request), None)
    return None, None, None, "", None, None


# En la aplicación de escritorio (escritorio.py) cerrar consiste en cerrar su ventana.
AL_CERRAR = None


def cerrar_aplicacion():
    """Detiene el servidor de forma ordenada: equivale a pulsar Ctrl + C en la terminal
    (Gradio cierra el servidor y el programa termina). Se hace con un pequeño retraso
    para que el navegador reciba antes la respuesta."""
    if AL_CERRAR is not None:
        threading.Timer(0.8, AL_CERRAR).start()
    else:
        threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    return gr.update(value="Cerrando…", interactive=False)


AVISO_CIERRE = """() => {
    setTimeout(() => {
        document.body.innerHTML = '<div style="font-family:sans-serif;max-width:32rem;margin:20vh auto;'
            + 'text-align:center;color:#333"><h2>Fotos Minis se ha cerrado</h2>'
            + '<p>Ya puedes cerrar esta pestaña. Para volver a usarla, abre de nuevo el lanzador.</p></div>';
    }, 1500);
}"""


# Aplica la apariencia en el navegador. Gradio usa la clase "dark" en <body>. En
# modo "sistema" sigue además los cambios del sistema operativo en caliente.
APLICAR_APARIENCIA = """(modo) => {
    const medio = window.matchMedia('(prefers-color-scheme: dark)');
    window.__fmModo = modo;
    const aplicar = () => {
        const m = window.__fmModo;
        document.body.classList.toggle('dark', m === 'nocturno' || (m === 'sistema' && medio.matches));
    };
    if (!window.__fmEscucha) { medio.addEventListener('change', aplicar); window.__fmEscucha = true; }
    aplicar();
    return modo;
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
/* Con una foto cargada Gradio oculta su zona de soltar y, al arrastrar otra encima,
   algunos navegadores (Safari) la vacían sin recoger la nueva. La zona de soltar se
   mantiene activa como una capa invisible sobre la foto: soltar otra la sustituye por
   el mismo camino que la primera vez, y un clic abre el selector de ficheros. */
#entrada .upload-container { position: relative; }
#entrada button.upload-container.hidden {
  display: flex !important; position: absolute !important; inset: 0 !important;
  width: 100% !important; height: 100% !important; opacity: 0 !important; z-index: 5;
}
#entrada .top-panel { z-index: 10; }
/* Pestañas del panel compactas, para que quepan las cinco sin menú "⋯". */
#panel .tab-wrapper button { padding: 6px 8px !important; font-size: 13px !important; }
#fila-carpeta { align-items: flex-end; gap: 8px; }
#fila-carpeta > button { height: 38px; margin-bottom: 2px; }
#guardado p { margin: 4px 0 0; font-size: 13px; opacity: .8; text-align: center; }
#apariencia { width: 330px !important; min-width: 330px !important; flex: none !important; }
#apariencia .wrap { flex-wrap: nowrap; gap: 6px; }
#apariencia label { padding: 6px 10px !important; white-space: nowrap; }
#visor-tabs .tabitem { padding: 0 !important; }
#comparador, #mascara, #galeria, #galeria-ind { height: calc(100vh - 405px) !important; min-height: 360px; }
/* Las imágenes ocupan todo el alto disponible del visor, sin deformarse. */
#comparador .content, #comparador .content img,
#mascara .image-container, #mascara img { height: 100% !important; max-height: none !important; }
#comparador img, #mascara img { width: 100% !important; object-fit: contain; }
@media (max-width: 960px) {
  #cabecera { flex-wrap: wrap !important; }
  #titulo { flex: 1 1 100% !important; min-width: 100% !important; }
  #panel { position: static; max-height: none; overflow: visible; }
  #comparador, #mascara, #galeria, #galeria-ind { height: 70vh !important; }
}
"""

with gr.Blocks(title="Fotos Minis · Fondo negro de estudio", fill_width=True) as demo:
    with gr.Row(elem_id="cabecera", equal_height=True):
        gr.Markdown(
            "# Fotos Minis · Fondo negro de estudio\n"
            "Sustituye el fondo por un negro de estudio conservando intactos la figura y la peana.",
            elem_id="titulo")
        apariencia = gr.Radio([(v, k) for k, v in preferencias.APARIENCIAS.items()],
                              value="sistema", label="Apariencia", scale=0, min_width=320,
                              elem_id="apariencia", container=False)
        boton_cerrar = gr.Button("Cerrar aplicación", variant="stop", scale=0, min_width=180,
                                 elem_id="boton-cerrar")

    with gr.Tabs():
        with gr.Tab("Una foto", id="una"):
            with gr.Row(equal_height=False):
                # ---------------- Panel de ajustes ----------------
                with gr.Column(scale=1, min_width=330, elem_id="panel"):
                    entrada = gr.Image(label="Foto original (suelta otra encima para cambiarla)", type="filepath", height=210, elem_id="entrada",
                                       sources=["upload", "clipboard"])
                    boton = gr.Button("Volver a procesar", variant="primary", size="sm")
                    with gr.Tabs():
                        with gr.Tab("Perfiles", id="perfiles"):
                            perfil = gr.Dropdown([FABRICA, *perfiles.listar()], value=FABRICA,
                                                 label="Perfil guardado", interactive=True)
                            with gr.Row():
                                b_aplicar = gr.Button("Aplicar", variant="primary", size="sm", min_width=60)
                                b_al_abrir = gr.Button("Usar al abrir", size="sm", min_width=60)
                                b_borrar = gr.Button("Borrar", variant="stop", size="sm", min_width=60)
                            gr.Markdown("---")
                            nombre_perfil = gr.Textbox(label="Guardar los ajustes actuales como…",
                                                       placeholder="p. ej. Peanas de arena con horizonte")
                            con_al_abrir = gr.Checkbox(True, label="Usar este perfil al abrir la aplicación")
                            b_guardar = gr.Button("Guardar perfil", size="sm")
                            estado_perfiles = gr.Markdown()
                        with gr.Tab("Recorte", id="recorte"):
                            modelo = gr.Dropdown([(v, k) for k, v in MODELOS.items()],
                                                 value=MODELO_POR_DEFECTO, label="Modelo de segmentación")
                            huecos = gr.Slider(0, 5000, 0, step=50, label="Rellenar huecos de la máscara (px)",
                                               info="Si aparecen agujeros en zonas oscuras de la figura")
                            descontaminar = gr.Checkbox(True, label="Limpiar el color del fondo antiguo en el contorno")
                            nitidez = gr.Slider(0, 1, 0.0, step=0.05, label="Corregir desenfoque por distancia",
                                                info="Mide cuánto está desenfocada cada zona de la figura (profundidad "
                                                     "de campo) y la enfoca en esa medida. 0 = figura intacta")
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
                        with gr.Tab("Fondo", id="fondo"):
                            suelo = gr.Radio([(v, k) for k, v in SUELOS.items()], value="estudio",
                                             label="Suelo")
                            with gr.Group(visible=False) as grupo_original:
                                gr.Markdown("Se conserva el suelo real de la foto y sólo la pared "
                                            "pasa a negro. El encuadre no sale de la foto.")
                                suelo_ajuste = gr.Slider(-0.5, 0.5, 0.0, step=0.01,
                                                         label="Ajuste de la línea suelo-pared",
                                                         info="+ la sube, − la baja (si la detección no es exacta)")
                            with gr.Group() as grupo_horizonte:
                                gr.Markdown("Paso del suelo a la pared, según la perspectiva de la peana; "
                                            "se desenfoca más cuanto más lejos queda la pared.")
                                horizonte = gr.Slider(0, 1, 0.0, step=0.01, label="Intensidad del horizonte",
                                                      info="Sólo suelo de estudio: 0 = negro continuo")
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
                                                     label="Proporción", info="Sin carrusel")
                            carrusel = gr.Radio([("No", 0), ("Auto", CARRUSEL_AUTO), ("2 partes", 2), ("3 partes", 3),
                                                 ("4 partes", 4), ("5 partes", 5), ("6 partes", 6)],
                                                value=0, label="Carrusel de Instagram",
                                                info="Imagen N:1 dividida en N cuadrados que, al deslizar, "
                                                     "se ven como una sola imagen continua. «Auto» elige el número de "
                                                     "partes con el que las minis ocupan más área")
                            individuales = gr.Checkbox(False, label="Fotos individuales de cada figura",
                                                       info="Si hay varias figuras, además de la foto de grupo "
                                                            "se recorta una por figura con la misma proporción "
                                                            "(1:1 en carrusel)")
                            superres = gr.Radio([("No", 0), ("×2", 2), ("×4", 4)], value=0,
                                                label="Super-resolución (IA) de las fotos individuales",
                                                info="Desactivada por defecto. Más nitidez en los recortes "
                                                     "pequeños, pero la IA reinterpreta los detalles finos. "
                                                     "Tarda unos segundos por figura; máximo 4096 px")
                            superres_grupo = gr.Checkbox(False, label="Aplicar también a la foto de grupo y al carrusel",
                                                         info="Con ×2 o ×4. La foto «madre» suele ser grande: puede "
                                                              "tardar bastante y nunca pasa de 4096 px")

                # ---------------- Visor ----------------
                with gr.Column(scale=3, min_width=480):
                    with gr.Tabs(elem_id="visor-tabs"):
                        with gr.Tab("Antes / después", id="comparar"):
                            comparador = gr.ImageSlider(show_label=False, type="pil", elem_id="comparador")
                        with gr.Tab("Máscara", id="mascara_tab"):
                            mascara = gr.Image(show_label=False, type="pil", elem_id="mascara")
                        with gr.Tab("Carrusel", id="carrusel_tab"):
                            galeria = gr.Gallery(show_label=False, columns=4, object_fit="contain",
                                                 elem_id="galeria", allow_preview=True)
                        with gr.Tab("Individuales", id="individuales_tab"):
                            galeria_ind = gr.Gallery(show_label=False, columns=4, object_fit="contain",
                                                     elem_id="galeria-ind", allow_preview=True)
                    descarga = gr.DownloadButton("Descargar resultado (PNG sin pérdida; ZIP si hay carrusel o individuales)",
                                                 variant="primary")
                    guardado = gr.Markdown(elem_id="guardado")
                    with gr.Row(elem_id="fila-carpeta"):
                        carpeta = gr.Textbox(label="Carpeta de guardado (se recuerda para la próxima vez)",
                                             scale=5, max_lines=1, elem_id="carpeta",
                                             placeholder="Escribe o pega una ruta, o pulsa Elegir…")
                        b_elegir = gr.Button("Elegir…", scale=0, min_width=90, size="sm")
                        b_abrir = gr.Button("Abrir", scale=0, min_width=80, size="sm")

        with gr.Tab("Varias fotos", id="varias"):
            gr.Markdown("Se procesan con los ajustes elegidos en la pestaña **Una foto**, y el ZIP "
                        "se guarda en la carpeta de guardado indicada allí.")
            with gr.Row(equal_height=True):
                lote = gr.File(file_count="multiple", file_types=["image", ".heic", ".heif"],
                               label="Fotos", scale=2)
                with gr.Column(scale=1, min_width=260):
                    boton_lote = gr.Button("Procesar todas", variant="primary")
                    zip_salida = gr.File(label="ZIP con los resultados")
                    guardado_lote = gr.Markdown()

    ajustes = [halo, vineteado, sombra, reflejo, caida, oscurecer, dominante, negro, reencuadrar,
               margen, proporcion, descontaminar, suelo, suelo_ajuste, horizonte, h_distancia, h_desenfoque,
               h_suelo, h_altura, carrusel, individuales, superres, superres_grupo, nitidez]
    entradas = [entrada, modelo, huecos, *ajustes]
    salidas = [comparador, mascara, descarga, guardado, galeria, galeria_ind]
    boton.click(procesar_desde_ui, entradas, salidas)
    entrada.change(procesar_desde_ui, entradas, salidas)
    entrada.clear(limpiar, None, salidas)
    # "always_last": al arrastrar un control sólo se procesa el último valor, y
    # también reacciona cuando se escribe el número en la casilla.
    for c in [*ajustes, modelo, huecos]:
        c.change(procesar_desde_ui, entradas, salidas, trigger_mode="always_last", show_progress="minimal")
    controles = [modelo, huecos, *ajustes]  # mismo orden que perfiles.CAMPOS
    suelo.change(lambda v: (gr.update(visible=v == "original"), gr.update(visible=v != "original")),
                 suelo, [grupo_original, grupo_horizonte], queue=False)
    demo.load(al_abrir, None, [*controles, perfil, estado_perfiles, apariencia, carpeta]) \
        .then(None, apariencia, None, js=APLICAR_APARIENCIA)
    # Por la cola (no queue=False): sólo así Gradio muestra los avisos gr.Info/gr.Warning.
    carpeta.submit(cambiar_carpeta, carpeta, carpeta)
    carpeta.blur(cambiar_carpeta, carpeta, carpeta)
    b_elegir.click(elegir_carpeta, carpeta, carpeta)
    b_abrir.click(abrir_carpeta, None, None, queue=False)
    apariencia.input(guardar_apariencia, apariencia, None, js=APLICAR_APARIENCIA, queue=False)
    b_aplicar.click(aplicar_perfil, perfil, [*controles, estado_perfiles]) \
        .then(procesar_desde_ui, entradas, salidas)
    b_guardar.click(guardar_perfil, [nombre_perfil, con_al_abrir, *controles],
                    [perfil, nombre_perfil, estado_perfiles])
    nombre_perfil.submit(guardar_perfil, [nombre_perfil, con_al_abrir, *controles],
                         [perfil, nombre_perfil, estado_perfiles])
    b_al_abrir.click(usar_al_abrir, perfil, estado_perfiles)
    b_borrar.click(borrar_perfil, perfil, [perfil, estado_perfiles])
    boton_cerrar.click(cerrar_aplicacion, None, boton_cerrar, js=AVISO_CIERRE, queue=False)
    boton_lote.click(procesar_lote, [lote, modelo, huecos, *ajustes], [zip_salida, guardado_lote])


if __name__ == "__main__":
    # Si se arranca en segundo plano (p. ej. desde el lanzador), el shell hace que
    # se ignore Ctrl + C; se restaura para que el botón "Cerrar aplicación" funcione.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    # Las texturas de suelo se generan en segundo plano para que el primer uso sea inmediato.
    from fotos_minis.suelo import precargar
    threading.Thread(target=precargar, daemon=True).start()
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
