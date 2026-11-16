"""Pruebas sin red ni modelos: se usa una máscara sintética."""

import numpy as np

from fotos_minis.composicion import OpcionesEstudio, componer, geometria_peana
from fotos_minis.proceso import procesar
from fotos_minis.segmentacion import refinar_mascara


def _escena():
    rng = np.random.default_rng(0)
    h, w = 300, 240
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (200, 180, 120)  # fondo amarillento
    alfa = np.zeros((h, w), np.float32)
    # "Figura": rectángulo con textura y "peana": elipse en la base.
    yy, xx = np.mgrid[0:h, 0:w]
    figura = (xx > 90) & (xx < 150) & (yy > 40) & (yy < 220)
    peana = ((xx - 120) / 60) ** 2 + ((yy - 230) / 15) ** 2 <= 1
    sujeto = figura | peana
    img[sujeto] = rng.integers(0, 255, (sujeto.sum(), 3))
    alfa[sujeto] = 1.0
    return img, alfa


def test_sujeto_intacto_y_fondo_negro():
    img, alfa = _escena()
    op = OpcionesEstudio(reencuadrar=False, halo=0, vineteado=0, sombra=0, oscurecer_borde=0)
    out = componer(img, alfa, op)
    nucleo = alfa >= 1.0
    assert np.array_equal(out[nucleo], img[nucleo])
    assert out[alfa == 0].max() == 0


def test_reencuadre_no_reescala():
    img, alfa = _escena()
    op = OpcionesEstudio(proporcion="1:1", margen=0.2, oscurecer_borde=0)
    out = componer(img, alfa, op)
    assert out.shape[0] == out.shape[1]
    # El parche del sujeto aparece idéntico en algún lugar del lienzo.
    parche = img[100:150, 100:140]
    encontrado = any(
        np.array_equal(out[y:y + 50, x:x + 40], parche)
        for y in range(out.shape[0] - 50) for x in range(out.shape[1] - 40))
    assert encontrado


def test_geometria_peana():
    _, alfa = _escena()
    g = geometria_peana(alfa)
    assert abs(g["cx"] - 120) < 3
    assert 110 <= g["ancho"] <= 125
    assert 240 <= g["y_suelo"] <= 246


def test_refinado_elimina_islas():
    img, alfa = _escena()
    alfa[5:12, 5:12] = 1.0  # mota de ruido
    r = refinar_mascara(img, alfa)
    assert r[5:12, 5:12].max() == 0
    assert r[100:200, 100:140].min() == 1.0


def test_procesar_con_mascara_dada():
    from PIL import Image

    img, alfa = _escena()
    r = procesar(Image.fromarray(img), OpcionesEstudio(reflejo=0.4), alfa=alfa)
    assert r.estudio.mode == "RGB" and r.recorte.mode == "RGBA"


def test_reencuadre_centra_en_horizontal_y_vertical():
    from fotos_minis.composicion import _pegar, caja_sujeto, calcular_lienzo

    img, alfa = _escena()
    # Sujeto pegado a la esquina de la foto y con una penumbra débil hasta el borde.
    alfa = np.roll(alfa, (40, -70), axis=(0, 1))
    alfa[-3:, 100:140] = 0.2
    for op in [OpcionesEstudio(), OpcionesEstudio(proporcion="16:9"), OpcionesEstudio(reflejo=0.5)]:
        ox, oy, W, H = calcular_lienzo(alfa, op)
        lienzo = np.zeros((H, W), np.float32)
        _pegar(lienzo, alfa, ox, oy)
        x0, y0, x1, y1 = caja_sujeto(lienzo, 0.5)
        assert abs(x0 - (W - x1)) <= 1 and abs(y0 - (H - y1)) <= 1


def test_horizonte_coherente_con_la_perspectiva():
    from fotos_minis.composicion import geometria_peana, linea_horizonte

    _, alfa = _escena()  # peana: elipse de semiejes 60 x 15 -> inclinación 0,25
    g = geometria_peana(alfa)
    assert 0.18 <= g["inclinacion"] <= 0.32
    cerca, s1 = linea_horizonte(g, OpcionesEstudio(horizonte=1, horizonte_distancia=1))
    lejos, s2 = linea_horizonte(g, OpcionesEstudio(horizonte=1, horizonte_distancia=5))
    assert lejos < cerca < g["y_suelo"]   # más lejos -> más arriba en la imagen
    assert s2 > s1                        # más lejos -> más desenfocada
    out = componer(*_escena(), OpcionesEstudio(horizonte=1, reencuadrar=False, vineteado=0))
    assert out[5, 5].sum() < out[-5, 5].sum()  # pared más oscura que el suelo
