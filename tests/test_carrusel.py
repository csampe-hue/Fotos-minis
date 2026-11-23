"""Modo carrusel de Instagram: imagen N:1 dividida en N cuadrados contiguos."""

import numpy as np
import pytest
from PIL import Image

from fotos_minis.carrusel import dividir
from fotos_minis.composicion import OpcionesEstudio, caja_sujeto, calcular_lienzo, componer


def _grupo(h=300, w=420):
    img = np.full((h, w, 3), 120, np.uint8)
    alfa = np.zeros((h, w), np.float32)
    for cx in (110, 210, 310):                       # tres minis en fila
        alfa[120:230, cx - 25:cx + 25] = 1
    img[alfa > 0] = (200, 160, 60)
    return img, alfa


@pytest.mark.parametrize("n", [2, 3, 4])
def test_partes_cuadradas_y_continuas(n):
    img, alfa = _grupo()
    out = componer(img, alfa, OpcionesEstudio(carrusel=n))
    H, W = out.shape[:2]
    assert W == n * H                                         # proporción N:1 exacta
    partes = dividir(Image.fromarray(out), n)
    assert all(p.size == (H, H) for p in partes)              # cada parte 1:1
    juntas = np.hstack([np.asarray(p) for p in partes])
    assert np.array_equal(juntas, out)                        # continuas: ni un píxel perdido


@pytest.mark.parametrize("n", [2, 3, 4])
def test_las_minis_caben_enteras_y_ocupan_lo_maximo(n):
    img, alfa = _grupo()
    op = OpcionesEstudio(carrusel=n, margen=0.05)
    ox, oy, W, H = calcular_lienzo(alfa, op)
    x0, y0, x1, y1 = caja_sujeto(alfa, 0.5)
    bw, bh = x1 - x0, y1 - y0
    assert ox + x0 >= 0 and oy + y0 >= 0 and ox + x1 <= W and oy + y1 <= H   # enteras
    m = int(0.05 * max(bw, bh))
    # El lienzo es el mínimo N:1 que contiene al sujeto con su margen: lo toca en un eje.
    assert H == bh + 2 * m or W == bw + 2 * m or W == n * (bh + 2 * m)


def test_tamano_incorrecto_da_error():
    with pytest.raises(ValueError):
        dividir(Image.new("RGB", (500, 200)), 2)


def test_app_guarda_completa_y_partes_y_limpia_las_anteriores(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setenv("FOTOS_MINIS_SALIDA", str(tmp_path / "fotos"))
    monkeypatch.setenv("FOTOS_MINIS_PERFILES", str(tmp_path / "perfiles.json"))
    import app
    from fotos_minis import perfiles
    app._ultimas_firmas.clear()
    img, alfa = _grupo()
    ruta = tmp_path / "grupo.png"
    Image.fromarray(img).save(ruta)
    app._cache_mascaras[app._clave(img, "combinado", 0)] = (alfa, img.astype(np.float32) / 255)

    def procesar(n):
        v = perfiles.valores_fabrica() | {"carrusel": n}
        return app.procesar_una(str(ruta), "combinado", 0, *[v[k] for k in perfiles.CAMPOS_ESTUDIO],
                                progress=lambda *a, **k: None)

    def ficheros():
        return sorted(f.name for f in (tmp_path / "fotos").iterdir())

    r = procesar(3)
    assert ficheros() == ["grupo_carrusel_1de3.png", "grupo_carrusel_2de3.png",
                          "grupo_carrusel_3de3.png", "grupo_estudio.png"]
    assert r[2].endswith(".zip") and len(r[4]) == 3
    completa = Image.open(tmp_path / "fotos" / "grupo_estudio.png")
    assert completa.width == 3 * completa.height                # la imagen antes de cortar
    procesar(2)                                                  # otra prueba: 2 partes
    assert ficheros() == ["grupo_carrusel_1de2.png", "grupo_carrusel_2de2.png", "grupo_estudio.png"]
    procesar(0)                                                  # sin carrusel
    assert ficheros() == ["grupo_estudio.png"]


def test_partes_recomendadas():
    from fotos_minis.carrusel import partes_recomendadas
    assert partes_recomendadas(200, 100, 0) == 2
    assert partes_recomendadas(300, 100, 0) == 3
    assert partes_recomendadas(500, 100, 0) == 4
    assert partes_recomendadas(100, 200, 0) == 2      # figura alta: lo menos malo son 2
