"""Fotos individuales recortadas de la foto de grupo procesada."""

import numpy as np
import pytest
from PIL import Image

from fotos_minis.composicion import OpcionesEstudio, componer
from fotos_minis.individuales import figuras, individuales, localizar_peanas, recortes


def _grupo(posiciones, h=360, w=640):
    """Minis sintéticas: peana elíptica (60×16 de semiejes) y cuerpo encima."""
    img = np.full((h, w, 3), 120, np.uint8)
    alfa = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in posiciones:
        cuerpo = (np.abs(xx - cx) < 14) & (yy > cy - 110) & (yy < cy)
        cabeza = ((xx - cx) / 12) ** 2 + ((yy - (cy - 118)) / 12) ** 2 <= 1
        peana = ((xx - cx) / 60) ** 2 + ((yy - cy) / 16) ** 2 <= 1
        alfa[cuerpo | cabeza | peana] = 1
    img[alfa > 0] = (200, 160, 60)
    return img, alfa


def test_separa_peanas_que_se_tocan_y_solapan():
    # Fila delantera tocándose y una detrás asomando entre ellas.
    pos = [(150, 300), (270, 300), (390, 300), (210, 262), (330, 262)]
    _, alfa = _grupo(pos)
    centros = sorted((round(cx), round(cy)) for cx, cy, _, _ in localizar_peanas(alfa))
    assert len(centros) == 5
    for (cx, cy), (ex, ey) in zip(centros, sorted(pos)):
        assert abs(cx - ex) <= 6 and abs(cy - ey) <= 6


def test_caja_incluye_cabeza_y_peana():
    _, alfa = _grupo([(200, 300), (440, 300)])
    f = figuras(alfa)
    assert len(f) == 2
    x0, y0, x1, y1 = f[0].caja
    assert y0 <= 300 - 118 - 10 and y1 >= 300 + 14          # cabeza y borde inferior de la peana
    assert x0 <= 200 - 55 and x1 >= 200 + 55


@pytest.mark.parametrize("op, prop", [(OpcionesEstudio(proporcion="4:5"), 0.8),
                                      (OpcionesEstudio(carrusel=2), 1.0),
                                      (OpcionesEstudio(proporcion="1:1"), 1.0)])
def test_individuales_con_la_proporcion_objetivo(op, prop):
    img, alfa = _grupo([(180, 300), (320, 300), (460, 300)])
    out = componer(img, alfa, op)
    lista, tapadas = individuales(out, alfa, op, img)
    assert len(lista) == 3 and tapadas == 0
    for t, _ in lista:
        assert abs(t.shape[1] / t.shape[0] - prop) < 0.02
        assert (t.reshape(-1, 3) == (200, 160, 60)).all(axis=1).any()   # contiene una mini


def _mini_en(t):
    """Caja de los píxeles de mini dentro de una foto individual."""
    ys, xs = np.nonzero((t == (200, 160, 60)).all(axis=2))
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_brazo_levantado_entero_y_figura_centrada():
    img, alfa = _grupo([(170, 300), (330, 300), (490, 300)])
    yy, xx = np.mgrid[0:360, 0:640]
    brazo = (xx > 340) & (xx < 352) & (yy > 120) & (yy < 200)     # brazo alzado de la del medio
    alfa[brazo] = 1
    img[brazo] = (200, 160, 60)
    op = OpcionesEstudio(proporcion="1:1")
    lista, _ = individuales(componer(img, alfa, op), alfa, op, img)
    t = lista[1][0]
    x0, y0, x1, y1 = _mini_en(t)
    assert y1 - y0 >= (300 + 16) - 120 - 6       # de la mano a la peana (salvo el borde oscurecido)
    H, W = t.shape[:2]
    assert abs((x0 + x1) / 2 - W / 2) <= 0.06 * W                  # centrada en horizontal
    assert abs((y0 + y1) / 2 - H / 2) <= 0.06 * H                  # y en vertical


def test_figuras_lejanas_se_recortan_y_cercanas_llevan_fondo_nuevo():
    op = OpcionesEstudio(proporcion="1:1", margen=0.05)
    img, alfa = _grupo([(110, 300), (530, 300)], w=640)           # muy separadas
    lista, _ = individuales(componer(img, alfa, op), alfa, op, img)
    assert [m for _, m in lista] == ["recorte", "recorte"]
    img, alfa = _grupo([(250, 300), (380, 300)], w=640)           # pegadas: asomaría la vecina
    lista, _ = individuales(componer(img, alfa, op), alfa, op, img)
    assert [m for _, m in lista] == ["nuevo fondo", "nuevo fondo"]
    for t, _ in lista:                                            # y sin rastro de la vecina
        assert (t == (200, 160, 60)).all(axis=2).sum() <= 1.05 * (alfa[:, :320] > 0).sum()


def test_recorte_es_un_trozo_exacto_de_la_salida():
    img, alfa = _grupo([(110, 300), (530, 300)], w=640)
    op = OpcionesEstudio(proporcion="1:1", margen=0.05)
    out = componer(img, alfa, op)
    t = individuales(out, alfa, op, img)[0][0][0]
    h, w = t.shape[:2]
    assert any(np.array_equal(out[y:y + h, x:x + w], t)
               for y in range(out.shape[0] - h + 1) for x in range(out.shape[1] - w + 1)
               if (out[y, x] == t[0, 0]).all() and (out[y + h - 1, x + w - 1] == t[-1, -1]).all())


def test_una_sola_figura_no_genera_individuales():
    img, alfa = _grupo([(320, 300)])
    op = OpcionesEstudio(individuales=True)
    assert individuales(componer(img, alfa, op), alfa, op, img) == ([], 0)


def test_app_guarda_grupo_e_individuales(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setenv("FOTOS_MINIS_SALIDA", str(tmp_path / "fotos"))
    monkeypatch.setenv("FOTOS_MINIS_PERFILES", str(tmp_path / "perfiles.json"))
    import app
    from fotos_minis import perfiles
    app._ultimas_firmas.clear()
    img, alfa = _grupo([(180, 300), (320, 300), (460, 300)])
    ruta = tmp_path / "patrulla.png"
    Image.fromarray(img).save(ruta)
    app._cache_mascaras[app._clave(img, "combinado", 0)] = (alfa, img.astype(np.float32) / 255)

    def procesar(**c):
        v = perfiles.valores_fabrica() | c
        return app.procesar_una(str(ruta), "combinado", 0, *[v[k] for k in perfiles.CAMPOS_ESTUDIO],
                                progress=lambda *a, **k: None)

    def ficheros():
        return sorted(f.name for f in (tmp_path / "fotos").iterdir())

    r = procesar(individuales=True, carrusel=-1)
    assert [f for f in ficheros() if "figura" in f] == ["patrulla_figura1.png", "patrulla_figura2.png",
                                                       "patrulla_figura3.png"]
    assert "patrulla_estudio.png" in ficheros() and any("carrusel" in f for f in ficheros())
    assert len(r[5]) == 3 and r[2].endswith(".zip")
    for i in (1, 2, 3):
        im = Image.open(tmp_path / "fotos" / f"patrulla_figura{i}.png")
        assert im.width == im.height                       # carrusel -> 1:1
    procesar(individuales=False)                           # otra prueba sin individuales
    assert not any("figura" in f for f in ficheros())


def test_torso_ancho_no_cuenta_como_peana():
    """Figura corriendo: torso + pierna en horizontal parecen una peana muy por encima
    de la real, casi en su vertical y unida a ella -> se descarta."""
    from fotos_minis.individuales import _descartar_falsas
    alfa = np.zeros((400, 400), np.float32)
    alfa[100:330, 180:220] = 1                                   # una sola pieza
    reales = [(200, 320, 60, 16)]
    falsa = (205, 200, 60, 16)                                   # 7,5 alturas de peana más arriba
    tapada = (250, 296, 60, 16)                                  # detrás: 1,5 alturas más arriba
    assert _descartar_falsas(reales + [falsa, tapada], alfa) == reales + [tapada]


def test_superresolucion_de_las_individuales():
    from fotos_minis.superres import disponible, factor_efectivo
    if not disponible():
        pytest.skip("modelo de super-resolución no incluido")
    img, alfa = _grupo([(110, 300), (530, 300)], w=640)
    base = OpcionesEstudio(proporcion="1:1", margen=0.05)
    out = componer(img, alfa, base)
    normal = individuales(out, alfa, base, img)[0]
    x2 = individuales(out, alfa, OpcionesEstudio(proporcion="1:1", margen=0.05, superres=2), img)[0]
    for (n, _), (s, _) in zip(normal, x2):
        assert s.shape[0] == 2 * n.shape[0] and s.shape[1] == 2 * n.shape[1]
    assert factor_efectivo((3000, 3000), 4) == 4096 / 3000       # tope de 4096 px


def test_superres_amplia_sin_costuras():
    from fotos_minis.superres import TESELA, ampliar, disponible
    if not disponible():
        pytest.skip("modelo de super-resolución no incluido")
    yy, xx = np.mgrid[0:300, 0:300]
    img = np.dstack([(xx * 0.8).astype(np.uint8), (yy * 0.8).astype(np.uint8), np.full((300, 300), 90, np.uint8)])
    r = ampliar(img, 4).astype(float)
    assert r.shape == (1200, 1200, 3)
    d = np.abs(np.diff(r.mean(axis=2), axis=1)).mean(axis=0)
    assert d[4 * TESELA - 1] < 3 * np.median(d) + 1           # sin escalón en la costura


# --------------------------------------------------------------------------- #
# Casos difíciles para localizar peanas
# --------------------------------------------------------------------------- #

def _mini(alfa, cx, cy, a=60, b=16, alto=110):
    yy, xx = np.mgrid[0:alfa.shape[0], 0:alfa.shape[1]]
    cuerpo = (np.abs(xx - cx) < 14) & (yy > cy - alto) & (yy < cy)
    cabeza = ((xx - cx) / 12) ** 2 + ((yy - (cy - alto - 8)) / 12) ** 2 <= 1
    peana = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
    alfa[cuerpo | cabeza | peana] = 1
    return alfa


def _centros(alfa):
    return sorted((round(cx), round(cy)) for cx, cy, _, _ in localizar_peanas(alfa))


def _cerca(centros, esperados, tol=12):
    return len(centros) == len(esperados) and all(
        abs(a - c) <= tol and abs(b - d) <= tol for (a, b), (c, d) in zip(centros, sorted(esperados)))


def test_lanza_que_une_dos_figuras():
    """Una lanza horizontal une dos figuras separadas en una sola pieza de la silueta:
    antes se perdía una peana y las otras salían desplazadas."""
    import cv2
    pos = [(200, 330), (360, 330), (520, 330)]
    alfa = np.zeros((420, 720), np.float32)
    for p in pos:
        _mini(alfa, *p)
    cv2.line(alfa, (200, 300), (360, 300), 1, 6)
    assert _cerca(_centros(alfa), pos)
    assert len(figuras(alfa)) == 3


def test_capa_ancha_no_crea_peana_fantasma():
    """Una capa que se abre hasta tocar la peana parece otra peana de detrás."""
    import cv2
    alfa = _mini(np.zeros((420, 720), np.float32), 360, 330)
    cv2.fillPoly(alfa, [np.array([[345, 210], [375, 210], [450, 322], [270, 322]], np.int32)], 1)
    assert _cerca(_centros(alfa), [(360, 330)])
    for p in [(200, 330), (520, 330)]:
        _mini(alfa, *p)
    assert _cerca(_centros(alfa), [(200, 330), (360, 330), (520, 330)])


def test_peanas_de_tamanos_distintos():
    """Peana de 50 mm entre dos de 25 mm: cada pieza mide su propia peana (antes la
    grande se partía en tres pequeñas)."""
    alfa = np.zeros((420, 720), np.float32)
    _mini(alfa, 170, 330, 40, 11, 80)
    _mini(alfa, 340, 330, 80, 21, 140)
    _mini(alfa, 520, 330, 40, 11, 80)
    assert _cerca(_centros(alfa), [(170, 330), (340, 330), (520, 330)])


def test_fila_trasera_que_asoma_por_los_lados():
    """Dos delante y dos detrás desplazadas: el borde inferior no forma muescas sino
    escalones (donde acaba una peana delantera asoma la de detrás)."""
    pos = [(220, 340), (360, 340), (290, 300), (430, 300)]
    alfa = np.zeros((420, 720), np.float32)
    for p in pos:
        _mini(alfa, *p)
    assert _cerca(_centros(alfa), pos)
    import cv2                                                     # también en foto grande
    grande = cv2.resize(alfa, (2880, 1680), interpolation=cv2.INTER_LINEAR)
    assert _cerca([(round(x / 4), round(y / 4)) for x, y in _centros(grande)], pos)


def test_superres_desactivada_por_defecto():
    from fotos_minis.perfiles import valores_fabrica
    assert OpcionesEstudio().superres == 0
    assert valores_fabrica()["superres"] == 0


def test_superres_x2_por_teselas_igual_que_reducir_x4():
    """×2 se reduce tesela a tesela (sin guardar el ×4 entero) y da lo mismo que reducir
    la imagen ×4 completa."""
    import cv2
    from fotos_minis.superres import TESELA, _ampliar_ia, disponible
    if not disponible():
        pytest.skip("modelo de super-resolución no incluido")
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (TESELA + 37, TESELA + 11, 3), np.uint8), (5, 5), 0)
    x2 = _ampliar_ia(img, 2)
    ref = cv2.resize(_ampliar_ia(img, 4), (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_AREA)
    assert x2.shape == ref.shape
    assert np.abs(x2.astype(int) - ref.astype(int)).max() <= 1


def test_superres_imagen_grande_limita_la_memoria(monkeypatch):
    """Con el tope de LADO_MAXIMO se usa el factor entero más pequeño que llega a él."""
    import fotos_minis.superres as sr
    llamadas = []
    monkeypatch.setattr(sr, "disponible", lambda: True)
    monkeypatch.setattr(sr, "_ampliar_ia", lambda img, f: (llamadas.append(f),
                        np.zeros((img.shape[0] * f, img.shape[1] * f, 3), np.uint8))[1])
    r = sr.ampliar(np.zeros((300, 3000, 3), np.uint8), 4)        # tope: ×1,365 -> base ×2
    assert llamadas == [2] and r.shape == (410, 4096, 3)
    r = sr.ampliar(np.zeros((100, 1500, 3), np.uint8), 4)        # ×2,73 -> base ×4
    assert llamadas == [2, 4] and r.shape[1] == 4096
    img = np.zeros((100, 4000, 3), np.uint8)                     # ya es grande: sin cambios
    assert sr.ampliar(img, 4) is img and llamadas == [2, 4]


def test_superres_sin_modelo_sigue_y_avisa(monkeypatch):
    import fotos_minis.superres as sr
    monkeypatch.setattr(sr, "MODELO", sr.MODELO.with_name("no_existe.onnx"))
    with pytest.raises(sr.ErrorSuperres):
        sr.ampliar(np.zeros((50, 50, 3), np.uint8), 2)
    img, alfa = _grupo([(110, 300), (530, 300)], w=640)
    op = OpcionesEstudio(proporcion="1:1", margen=0.05, superres=2)
    out = componer(img, alfa, op)
    avisos = []
    con = individuales(out, alfa, op, img, avisos=avisos)[0]
    sin = individuales(out, alfa, OpcionesEstudio(proporcion="1:1", margen=0.05), img)[0]
    assert [a.shape for a, _ in con] == [s.shape for s, _ in sin]   # se guardan sin ampliar
    assert len(avisos) == 1 and "super-resolución" in avisos[0]      # un solo aviso


def test_superres_fallo_de_inferencia(monkeypatch):
    import fotos_minis.superres as sr

    def revienta(img, f):
        raise MemoryError
    monkeypatch.setattr(sr, "disponible", lambda: True)
    monkeypatch.setattr(sr, "_ampliar_ia", revienta)
    img = np.zeros((40, 40, 3), np.uint8)
    r, aviso = sr.ampliar_o_aviso(img, 4)
    assert r is img and "memoria" in aviso
