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
