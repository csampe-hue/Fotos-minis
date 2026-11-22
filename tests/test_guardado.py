"""Guardado automático en la carpeta «fotos» (sin modelos: la máscara se precarga)."""

import numpy as np
import pytest
from PIL import Image

gr = pytest.importorskip("gradio")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("FOTOS_MINIS_SALIDA", str(tmp_path / "fotos"))
    monkeypatch.setenv("FOTOS_MINIS_PERFILES", str(tmp_path / "perfiles.json"))
    import app as modulo
    modulo._ultimas_firmas.clear()
    return modulo


def _foto(tmp_path, nombre, color):
    img = np.full((120, 100, 3), color, np.uint8)
    img[30:100, 35:65] = (200, 60, 60)
    ruta = tmp_path / nombre
    Image.fromarray(img).save(ruta)
    alfa = np.zeros((120, 100), np.float32)
    alfa[30:100, 35:65] = 1
    return ruta, img, alfa


def _precargar(app, img, alfa, modelo="combinado", huecos=0):
    app._cache_mascaras[app._clave(img, modelo, huecos)] = (alfa, img.astype(np.float32) / 255)


def _ajustes(app, **cambios):
    from fotos_minis import perfiles
    v = perfiles.valores_fabrica() | cambios
    return [v[k] for k in perfiles.CAMPOS_ESTUDIO]


def _procesar(app, ruta, **cambios):
    return app.procesar_una(str(ruta), "combinado", 0, *_ajustes(app, **cambios),
                            progress=lambda *a, **k: None)


def test_varias_pruebas_con_la_misma_foto_dejan_solo_la_ultima(app, tmp_path):
    ruta, img, alfa = _foto(tmp_path, "mini.png", 120)
    _precargar(app, img, alfa)
    _procesar(app, ruta, halo=0.1)
    r = _procesar(app, ruta, halo=0.9, proporcion="1:1")
    fotos = sorted((tmp_path / "fotos").iterdir())
    assert [f.name for f in fotos] == ["mini_estudio.png"]
    guardada = Image.open(fotos[0])
    assert guardada.width == guardada.height            # es la última prueba (1:1)
    assert "mini_estudio.png" in r[3]


def test_otra_foto_original_se_guarda_aparte(app, tmp_path):
    for nombre, color in [("uno.png", 120), ("dos.png", 90)]:
        ruta, img, alfa = _foto(tmp_path, nombre, color)
        _precargar(app, img, alfa)
        _procesar(app, ruta)
    assert sorted(f.name for f in (tmp_path / "fotos").iterdir()) == ["dos_estudio.png", "uno_estudio.png"]


def test_quitar_la_foto_no_borra_lo_guardado(app, tmp_path):
    ruta, img, alfa = _foto(tmp_path, "mini.png", 120)
    _precargar(app, img, alfa)
    _procesar(app, ruta)
    app.limpiar()
    assert (tmp_path / "fotos" / "mini_estudio.png").exists()


def test_otra_pestana_con_la_misma_foto_tambien_obtiene_resultado(app, tmp_path):
    """Recargar la página y subir la misma foto no debe dejar el resultado vacío."""
    from types import SimpleNamespace
    ruta, img, alfa = _foto(tmp_path, "mini.png", 120)
    _precargar(app, img, alfa)
    a = app.procesar_una(str(ruta), "combinado", 0, *_ajustes(app), request=SimpleNamespace(session_hash="A"),
                         progress=lambda *x, **k: None)
    b = app.procesar_una(str(ruta), "combinado", 0, *_ajustes(app), request=SimpleNamespace(session_hash="B"),
                         progress=lambda *x, **k: None)
    repetida = app.procesar_una(str(ruta), "combinado", 0, *_ajustes(app), request=SimpleNamespace(session_hash="B"),
                                progress=lambda *x, **k: None)
    assert a[2] and b[2]                                   # las dos pestañas tienen resultado
    assert repetida[0] is not None and not isinstance(repetida[0], tuple)  # en la misma, se omite
