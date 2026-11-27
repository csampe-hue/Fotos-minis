"""Regresión de la detección de peanas y del recorte de cada figura.

Caso principal: una figura corriendo (torso y una pierna levantada en horizontal)
forma a la altura de la cadera una mancha ancha y plana que se parece a una peana.
Antes se tomaba como una segunda peana, se partía la figura en dos y la foto
individual de la figura central salía sin cabeza. Una peana real tapada por otra
queda como mucho ~1,7 alturas de peana por encima; la falsa queda mucho más arriba.
"""

import numpy as np
import pytest

from fotos_minis.composicion import OpcionesEstudio, componer
from fotos_minis.individuales import (_descartar_falsas, figuras, individuales, localizar_peanas,
                                      mascaras_figuras)

from test_individuales import _grupo, _mini_en

COLOR = (200, 160, 60)
CABEZA_CORREDOR = 300 - 152           # y de la parte de arriba de la cabeza del corredor


def _con_corredor(cx=320, cy=300, vecinas=((170, 300), (470, 300))):
    """Dos minis normales y, entre ellas, una corriendo: peana, pierna de apoyo,
    cadera + pierna en horizontal (elipse plana del ancho de una peana), torso y cabeza."""
    img, alfa = _grupo(list(vecinas))
    h, w = alfa.shape
    yy, xx = np.mgrid[0:h, 0:w]
    peana = ((xx - cx) / 60) ** 2 + ((yy - cy) / 16) ** 2 <= 1
    apoyo = (np.abs(xx - cx) < 7) & (yy > cy - 60) & (yy < cy)
    cadera = ((xx - cx) / 60) ** 2 + ((yy - (cy - 70)) / 14) ** 2 <= 1
    torso = (np.abs(xx - cx) < 12) & (yy > cy - 130) & (yy < cy - 60)
    cabeza = ((xx - cx) / 12) ** 2 + ((yy - (cy - 140)) / 12) ** 2 <= 1
    m = peana | apoyo | cadera | torso | cabeza
    alfa[m] = 1
    img[m] = COLOR
    return img, alfa


def _centros(alfa):
    return sorted((round(cx), round(cy)) for cx, cy, _, _ in localizar_peanas(alfa))


# --- Detección de peanas ----------------------------------------------------------

def test_pierna_horizontal_no_es_una_segunda_peana():
    _, alfa = _con_corredor()
    centros = _centros(alfa)
    assert len(centros) == 3
    for (cx, cy), (ex, ey) in zip(centros, [(170, 300), (320, 300), (470, 300)]):
        assert abs(cx - ex) <= 6 and abs(cy - ey) <= 6


def test_corredor_solo_con_una_vecina():
    _, alfa = _con_corredor(cx=250, vecinas=((450, 300),))
    assert len(_centros(alfa)) == 2


def test_peana_tapada_justo_detras_se_conserva():
    """Una peana real detrás de otra, casi en su vertical y unida a ella (~1,6 alturas
    de peana más arriba) no debe confundirse con la falsa del corredor."""
    _, alfa = _grupo([(170, 300), (320, 300), (340, 272), (470, 300)])
    centros = _centros(alfa)
    assert len(centros) == 4
    assert any(abs(cx - 340) <= 6 and abs(cy - 272) <= 6 for cx, cy in centros)


# --- Regla de descarte (_descartar_falsas) -----------------------------------------

def _pieza_unica():
    alfa = np.zeros((400, 400), np.float32)
    alfa[100:340, 150:250] = 1
    return alfa


@pytest.mark.parametrize("dy_en_b, se_queda", [(1.0, True), (1.5, True), (1.7, True), (2.0, True),
                                               (3.0, False), (5.0, False), (7.5, False)])
def test_umbral_de_altura(dy_en_b, se_queda):
    b = 16
    real = (200, 320, 60, b)
    otra = (205, 320 - dy_en_b * b, 60, b)
    assert (otra in _descartar_falsas([real, otra], _pieza_unica())) is se_queda
    assert real in _descartar_falsas([real, otra], _pieza_unica())       # la de abajo nunca


def test_no_descarta_si_esta_desplazada_en_horizontal():
    real, arriba = (200, 320, 60, 16), (240, 200, 60, 16)                   # |dx| = 40 > a/2
    assert _descartar_falsas([real, arriba], _pieza_unica()) == [real, arriba]


def test_no_descarta_si_no_estan_unidas():
    """Una figura en una fila de atrás (pieza separada) muy por encima es real."""
    alfa = np.zeros((400, 400), np.float32)
    alfa[100:215, 150:250] = 1
    alfa[260:340, 150:250] = 1
    real, atras = (200, 320, 60, 16), (205, 200, 60, 16)
    assert _descartar_falsas([real, atras], alfa) == [real, atras]


def test_no_descarta_si_el_centro_cae_fuera_de_la_silueta():
    alfa = np.zeros((400, 400), np.float32)
    alfa[300:340, 150:250] = 1
    real, vacia = (200, 320, 60, 16), (205, 200, 60, 16)
    assert _descartar_falsas([real, vacia], alfa) == [real, vacia]


# --- Máscaras y recorte de cada figura -----------------------------------------------

def test_mascara_del_corredor_incluye_la_cabeza():
    img, alfa = _con_corredor()
    ms = mascaras_figuras(alfa, img)
    assert len(ms) == 3
    ys, xs = np.nonzero(ms[1] > 0.5)
    assert ys.min() <= CABEZA_CORREDOR + 2 and ys.max() >= 300 + 14
    assert xs.min() <= 320 - 55 and xs.max() >= 320 + 55             # peana y pierna enteras


def test_caja_del_corredor_va_de_la_cabeza_a_la_peana():
    _, alfa = _con_corredor()
    f = figuras(alfa)
    assert len(f) == 3
    x0, y0, x1, y1 = f[1].caja
    assert y0 <= CABEZA_CORREDOR + 2 and y1 >= 300 + 14


@pytest.mark.parametrize("op", [OpcionesEstudio(proporcion="1:1"), OpcionesEstudio(proporcion="4:5"),
                                OpcionesEstudio(carrusel=3)])
def test_foto_individual_del_corredor_no_corta_la_cabeza(op):
    img, alfa = _con_corredor()
    lista, tapadas = individuales(componer(img, alfa, op), alfa, op, img)
    assert len(lista) == 3 and tapadas == 0
    t = lista[1][0]
    x0, y0, x1, y1 = _mini_en(t)
    escala = (x1 - x0) / 120                                          # ancho de la peana: 120 px
    assert (y1 - y0) / escala >= (300 + 16 - CABEZA_CORREDOR) - 8     # de la cabeza a la peana
    H, W = t.shape[:2]
    assert y0 > 0                                                     # con margen por arriba
    assert abs((y0 + y1) / 2 - H / 2) <= 0.06 * H                     # centrada en vertical
