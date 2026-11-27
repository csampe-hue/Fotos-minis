"""Fotos Minis como aplicación de escritorio: una ventana propia, sin navegador.

Lo usa «Fotos Minis.app» (ver ``fotos_minis/instalar_app.py``). Arranca el servidor
y lo muestra en una ventana nativa con pywebview (el motor de Safari en macOS). Al
cerrar la ventana, o con el botón «Cerrar aplicación», se detiene todo.

Si pywebview no está disponible o falla, se abre en el navegador como siempre.
"""

from __future__ import annotations

import os
import platform
import threading
import urllib.request
import webbrowser
from pathlib import Path

PUERTO = 7860
URL = f"http://127.0.0.1:{PUERTO}"
_RECURSOS = Path(__file__).resolve().parent / "recursos"
ICONO = next((p for p in (_RECURSOS / "icono_app.png", _RECURSOS / "icono.png") if p.exists()),
             _RECURSOS / "icono.png")


def responde(url: str = URL) -> bool:
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except OSError:
        return False


def arrancar_servidor():
    """Arranca la interfaz en segundo plano (sin abrir navegador). Devuelve el módulo app."""
    import app
    from fotos_minis.suelo import precargar

    threading.Thread(target=precargar, daemon=True).start()
    app.demo.queue(default_concurrency_limit=1).launch(
        css=app.CSS, head=app.ZOOM_PANTALLA_COMPLETA, server_port=PUERTO, inbrowser=False, prevent_thread_lock=True, quiet=True)
    return app


def _identidad_mac() -> None:
    """Nombre e icono de la aplicación en el Dock y la barra de menús de macOS."""
    try:
        from AppKit import NSApplication, NSImage
        from Foundation import NSBundle

        info = NSBundle.mainBundle().localizedInfoDictionary() or NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = "Fotos Minis"
        if ICONO.exists():
            imagen = NSImage.alloc().initWithContentsOfFile_(str(ICONO))
            NSApplication.sharedApplication().setApplicationIconImage_(imagen)
    except Exception:  # noqa: BLE001 - es sólo estético
        pass


def main() -> None:
    ya_abierta = responde()
    app = None if ya_abierta else arrancar_servidor()

    try:
        import webview
    except ImportError:
        webview = None

    if webview is None:
        webbrowser.open(URL)
        if app is not None:
            app.demo.block_thread()
        return

    if platform.system() == "Darwin":
        _identidad_mac()
    webview.settings["ALLOW_DOWNLOADS"] = True            # botón «Descargar resultado»
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    ventana = webview.create_window(
        "Fotos Minis", URL, width=1440, height=920, min_size=(900, 640),
        background_color="#111111", text_select=True)
    if app is not None:
        # El botón «Cerrar aplicación» cierra la ventana (y con ella, todo).
        app.AL_CERRAR = ventana.destroy
    almacen = Path.home() / ".fotos-minis" / "ventana"
    almacen.mkdir(parents=True, exist_ok=True)
    try:
        # private_mode=False: la ventana conserva lo que guarda la página (p. ej. el tema).
        webview.start(private_mode=False, storage_path=str(almacen))
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo abrir la ventana propia ({e}); se usa el navegador.")
        webbrowser.open(URL)
        if app is not None:
            app.demo.block_thread()
        return
    # Ventana cerrada.
    if app is not None:
        app.demo.close()
    os._exit(0)  # termina también los hilos en segundo plano


if __name__ == "__main__":
    main()
