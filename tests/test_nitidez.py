"""Corrección del desenfoque por distancia y super-resolución de la foto de grupo."""

import numpy as np
import pytest
from PIL import Image

import cv2

from fotos_minis.composicion import OpcionesEstudio, componer


def _textura(h=360, w=300, semilla=0):
    """Figura «pintada» (manchas y trazos de distintos tamaños) sobre fondo gris, y su máscara."""
    rng = np.random.default_rng(semilla)
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(140):
        color = tuple(int(c) for c in rng.integers(20, 235, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), int(rng.integers(6, 22)), color, -1)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-40, 40)), y + int(rng.integers(-40, 40))), color,
                     int(rng.integers(2, 6)))
    alfa = np.zeros((h, w), np.float32)
    alfa[30:h - 30, 40:w - 40] = 1
    img[alfa == 0] = 128
    return img, alfa


def _desenfoque_variable(img, arriba, abajo):
    """Desenfoque gaussiano que cambia con la altura (como la profundidad de campo)."""
    h = img.shape[0]
    sig = np.linspace(arriba, abajo, h)
    niveles = np.linspace(min(arriba, abajo), max(arriba, abajo), 16)
    pila = [cv2.GaussianBlur(img.astype(np.float32), (0, 0), max(s, 0.01)) for s in niveles]
    out = np.zeros_like(pila[0])
    idx = np.abs(sig[:, None] - niveles[None]).argmin(1)
    for y in range(h):
        out[y] = pila[idx[y]][y]
    return np.clip(np.round(out), 0, 255).astype(np.uint8), sig


def _psnr(a, b, m):
    return 10 * np.log10(255 ** 2 / np.mean((a[m].astype(float) - b[m].astype(float)) ** 2))


def test_el_mapa_sigue_al_desenfoque_real():
    from fotos_minis.nitidez import mapa
    img, alfa = _textura()
    borrosa, sig = _desenfoque_variable(img, 3.0, 0.5)
    m = mapa(borrosa, alfa)
    h = img.shape[0]
    arriba = m[40:h // 3][alfa[40:h // 3] > 0].mean()
    abajo = m[2 * h // 3:h - 40][alfa[2 * h // 3:h - 40] > 0].mean()
    assert arriba > abajo + 1.0                      # más desenfoque arriba que abajo
    assert abs(arriba - sig[40:h // 3].mean()) < 1.0  # y en la medida correcta (±1 px)


def test_restaura_mas_donde_mas_desenfocado_y_no_toca_el_fondo():
    from fotos_minis.nitidez import restaurar
    img, alfa = _textura()
    borrosa, _ = _desenfoque_variable(img, 3.0, 0.5)
    r = restaurar(borrosa, alfa, 1.0)
    h = img.shape[0]
    m = alfa > 0
    arriba = np.zeros_like(m)
    arriba[:h // 3] = m[:h // 3]
    assert _psnr(r, img, m) > _psnr(borrosa, img, m) + 0.5
    assert _psnr(r, img, arriba) > _psnr(borrosa, img, arriba) + 0.4
    # Fuera del sujeto (y de su borde) no cambia nada.
    lejos = cv2.dilate(alfa, np.ones((9, 9), np.uint8)) == 0
    assert np.array_equal(r[lejos], borrosa[lejos])
    assert np.array_equal(restaurar(borrosa, alfa, 0.0), borrosa)


def test_una_foto_nitida_casi_no_cambia():
    from fotos_minis.nitidez import restaurar
    img, alfa = _textura()
    r = restaurar(img, alfa, 1.0)
    assert _psnr(r, img, alfa > 0) > 28


def test_componer_con_nitidez():
    img, alfa = _textura()
    borrosa, _ = _desenfoque_variable(img, 3.0, 0.5)
    normal = componer(borrosa, alfa, OpcionesEstudio())
    nitida = componer(borrosa, alfa, OpcionesEstudio(nitidez=1.0))
    assert normal.shape == nitida.shape
    assert not np.array_equal(normal, nitida)
    assert OpcionesEstudio().nitidez == 0.0           # desactivada por defecto


def test_perfiles_sanean_las_opciones_nuevas():
    from fotos_minis.perfiles import a_opciones, valores_fabrica
    v = valores_fabrica()
    assert v["nitidez"] == 0.0 and v["superres_grupo"] is False
    op, _, _ = a_opciones(v | {"nitidez": 7, "superres_grupo": 1})
    assert op.nitidez == 1.0 and op.superres_grupo is True
    op, _, _ = a_opciones(v | {"nitidez": "x"})
    assert op.nitidez == 0.0


@pytest.fixture
def ia_falsa(monkeypatch):
    """Sustituye el modelo por un reescalado bicúbico (rápido y sin onnxruntime)."""
    from fotos_minis import superres
    monkeypatch.setattr(superres, "disponible", lambda: True)
    monkeypatch.setattr(superres, "_ampliar_ia",
                        lambda img, f: cv2.resize(img, (img.shape[1] * f, img.shape[0] * f),
                                                  interpolation=cv2.INTER_CUBIC))


def test_ampliar_grupo_mantiene_el_carrusel_exacto(ia_falsa):
    from fotos_minis.superres import LADO_MAXIMO, ampliar_grupo
    img = np.zeros((300, 900, 3), np.uint8)
    r, aviso = ampliar_grupo(img, 2, partes=3)
    assert aviso is None and r.shape == (600, 1800, 3)
    # Con el tope de 4096 px el factor no es entero: aun así, ancho = 3 × alto.
    img = np.zeros((701, 2103, 3), np.uint8)
    r, _ = ampliar_grupo(img, 2, partes=3)
    assert r.shape[1] == 3 * r.shape[0] and r.shape[1] <= LADO_MAXIMO
    # Ya demasiado grande: se deja igual y se avisa.
    img = np.zeros((1000, 4000, 3), np.uint8)
    r, aviso = ampliar_grupo(img, 2)
    assert r is img and "no se amplía" in aviso
    assert ampliar_grupo(img, 0) == (img, None)


def test_app_amplia_la_foto_madre_y_su_carrusel(tmp_path, monkeypatch, ia_falsa):
    pytest.importorskip("gradio")
    monkeypatch.setenv("FOTOS_MINIS_SALIDA", str(tmp_path / "fotos"))
    monkeypatch.setenv("FOTOS_MINIS_PERFILES", str(tmp_path / "perfiles.json"))
    import app
    from fotos_minis import perfiles
    app._ultimas_firmas.clear()
    h, w = 300, 420
    img = np.full((h, w, 3), 120, np.uint8)
    alfa = np.zeros((h, w), np.float32)
    for cx in (110, 210, 310):
        alfa[120:230, cx - 25:cx + 25] = 1
    img[alfa > 0] = (200, 160, 60)
    ruta = tmp_path / "grupo.png"
    Image.fromarray(img).save(ruta)
    app._cache_mascaras[app._clave(img, "combinado", 0)] = (alfa, img.astype(np.float32) / 255)

    def procesar(**c):
        v = perfiles.valores_fabrica() | c
        return app.procesar_una(str(ruta), "combinado", 0, *[v[k] for k in perfiles.CAMPOS_ESTUDIO],
                                progress=lambda *a, **k: None)

    procesar(carrusel=3)
    normal = Image.open(tmp_path / "fotos" / "grupo_estudio.png").size
    r = procesar(carrusel=3, superres=2, superres_grupo=True, individuales=True)
    grande = Image.open(tmp_path / "fotos" / "grupo_estudio.png").size
    assert grande == (2 * normal[0], 2 * normal[1])
    antes, despues = r[0]
    assert antes.size == despues.size == grande          # el comparador sigue alineado
    for i in (1, 2, 3):
        parte = Image.open(tmp_path / "fotos" / f"grupo_carrusel_{i}de3.png")
        assert parte.size == (grande[1], grande[1])
    # Las individuales se amplían una sola vez (×2), no ×2 sobre una madre ya ampliada.
    sin_sr = procesar(carrusel=3, individuales=True)[5]
    con_sr = r[5]
    for (a, _), (b, _) in zip(sin_sr, con_sr):
        assert Image.open(b).size == tuple(2 * x for x in Image.open(a).size)
