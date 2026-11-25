"""Fotos individuales recortadas de la foto de grupo procesada."""

import numpy as np
import pytest
from PIL import Image

from fotos_minis.composicion import OpcionesEstudio, componer
from fotos_minis.individuales import figuras, localizar_peanas, recortes


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
def test_recortes_con_la_proporcion_objetivo(op, prop):
    img, alfa = _grupo([(180, 300), (320, 300), (460, 300)])
    out = componer(img, alfa, op)
    trozos = recortes(out, alfa, op)
    assert len(trozos) == 3
    for t in trozos:
        assert abs(t.shape[1] / t.shape[0] - prop) < 0.02
        assert (t.reshape(-1, 3) == (200, 160, 60)).all(axis=1).any()   # contiene una mini


def test_recorte_es_un_trozo_exacto_de_la_salida():
    img, alfa = _grupo([(200, 300), (440, 300)])
    op = OpcionesEstudio(proporcion="1:1")
    out = componer(img, alfa, op)
    t = recortes(out, alfa, op)[0]
    h, w = t.shape[:2]
    encontrado = any(np.array_equal(out[y:y + h, x:x + w], t)
                     for y in range(out.shape[0] - h + 1) for x in range(0, out.shape[1] - w + 1, 1)
                     if out[y, x].tolist() == t[0, 0].tolist())
    assert encontrado                                        # recorte, sin reprocesar


def test_una_sola_figura_no_genera_individuales():
    img, alfa = _grupo([(320, 300)])
    op = OpcionesEstudio(individuales=True)
    assert recortes(componer(img, alfa, op), alfa, op) == []


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
