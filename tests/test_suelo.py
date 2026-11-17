"""Pruebas de los modos de suelo (escena sintética: pared clara arriba, tapete oscuro abajo)."""

import numpy as np

from fotos_minis.composicion import OpcionesEstudio, componer, mascara_suelo
from fotos_minis.suelo import detectar_suelo, textura

H, W, LINEA = 300, 400, 120   # la unión suelo-pared está en y = 120


def _escena(con_suelo=True):
    rng = np.random.default_rng(1)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = 170                                                    # pared gris clara
    if con_suelo:
        img[LINEA:] = 70                                            # tapete oscuro
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)  # textura
    alfa = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    figura = (xx > 180) & (xx < 220) & (yy > 60) & (yy < 230)
    peana = ((xx - 200) / 60) ** 2 + ((yy - 235) / 15) ** 2 <= 1
    sujeto = figura | peana
    img[sujeto] = (200, 60, 60)
    alfa[sujeto] = 1
    return img, alfa


def test_detecta_la_linea_suelo_pared():
    img, alfa = _escena()
    m = detectar_suelo(img, alfa)
    assert m is not None
    linea = (m > 0.5).argmax(axis=0)                 # primera fila de suelo por columna
    assert np.abs(np.median(linea) - LINEA) <= 4


def test_sin_suelo_distinguible_devuelve_none():
    img, alfa = _escena(con_suelo=False)
    assert detectar_suelo(img, alfa) is None


def test_suelo_original_se_conserva_y_la_pared_pasa_a_negro():
    img, alfa = _escena()
    out = componer(img, alfa, OpcionesEstudio(suelo="original", reencuadrar=False, halo=0, vineteado=0))
    assert np.array_equal(out[LINEA + 20:, :100], img[LINEA + 20:, :100])   # suelo intacto
    assert out[:LINEA - 20, :100].max() == 0                                # pared negra


def test_ajuste_mueve_la_linea():
    img, alfa = _escena()
    base = mascara_suelo(img, alfa, OpcionesEstudio(suelo="original"))
    sube = mascara_suelo(img, alfa, OpcionesEstudio(suelo="original", suelo_ajuste=0.1))
    assert (sube > 0.5).sum() > (base > 0.5).sum()


def test_sin_suelo_distinguible_cae_a_estudio():
    img, alfa = _escena(con_suelo=False)
    op = OpcionesEstudio(suelo="original", reencuadrar=False)
    assert np.array_equal(componer(img, alfa, op), componer(img, alfa, OpcionesEstudio(reencuadrar=False)))


def test_texturas_periodicas_y_suelo_bajo_el_horizonte():
    for tipo in ("oscuro", "arido"):
        t = textura(tipo)
        assert t.shape[2] == 3 and 0 <= t.min() and t.max() <= 1
        img, alfa = _escena()
        out = componer(img, alfa, OpcionesEstudio(suelo=tipo, reencuadrar=False, vineteado=0, halo=0,
                                                  oscurecer_borde=0))
        assert out[5:20].mean() < 3                       # pared negra arriba
        assert out[-30:, :60].mean() > 20                 # suelo con textura abajo
        assert out[-30:, :60].std() > 2                   # y no es plano
        nucleo = alfa >= 1
        assert np.array_equal(out[nucleo], img[nucleo])   # la figura sigue intacta
