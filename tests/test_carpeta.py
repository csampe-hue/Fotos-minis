"""Carpeta de guardado elegida por el usuario y recordada entre aperturas."""

import subprocess

import pytest

from fotos_minis import preferencias


@pytest.fixture(autouse=True)
def entorno(tmp_path, monkeypatch):
    monkeypatch.setenv("FOTOS_MINIS_PERFILES", str(tmp_path / "conf" / "perfiles.json"))
    monkeypatch.delenv("FOTOS_MINIS_SALIDA", raising=False)
    monkeypatch.chdir(tmp_path)


def test_por_defecto_es_fotos_junto_a_la_app(tmp_path):
    assert preferencias.carpeta_guardado() == tmp_path / "fotos"
    assert (tmp_path / "fotos").is_dir()


def test_se_recuerda_la_ultima_carpeta(tmp_path):
    elegida = preferencias.fijar_carpeta_guardado(str(tmp_path / "Mis minis" / "2026"))
    assert elegida.is_dir()
    assert preferencias.carpeta_guardado() == elegida          # «siguiente apertura»


def test_ruta_con_virgulilla_y_comillas(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert preferencias.fijar_carpeta_guardado('"~/Escritorio/minis"') == tmp_path / "Escritorio" / "minis"


def test_rutas_no_validas(tmp_path):
    fichero = tmp_path / "no_soy_carpeta.txt"
    fichero.write_text("x")
    with pytest.raises(ValueError):
        preferencias.fijar_carpeta_guardado(str(fichero))
    with pytest.raises(ValueError):
        preferencias.fijar_carpeta_guardado("   ")


def test_si_la_carpeta_recordada_ya_no_sirve_se_usa_la_de_por_defecto(tmp_path):
    preferencias.fijar_carpeta_guardado(str(tmp_path / "disco_externo"))
    (tmp_path / "disco_externo").rmdir()
    (tmp_path / "disco_externo").write_text("ahora es un fichero")   # ya no es utilizable
    assert preferencias.carpeta_guardado() == tmp_path / "fotos"


def test_selector_de_carpetas_del_mac(monkeypatch, tmp_path):
    llamadas = []

    def falso_run(args, **kw):
        llamadas.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="/Users/yo/Fotos minis/\n", stderr="")

    monkeypatch.setattr(preferencias.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(preferencias.subprocess, "run", falso_run)
    assert preferencias.elegir_carpeta_sistema(tmp_path) == "/Users/yo/Fotos minis/"
    assert llamadas[0][0] == "osascript" and any("choose folder" in a for a in llamadas[0])


def test_cancelar_el_selector_no_cambia_nada(monkeypatch, tmp_path):
    monkeypatch.setattr(preferencias.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(preferencias.subprocess, "run",
                        lambda args, **kw: subprocess.CompletedProcess(args, 1, stdout="", stderr="cancelado (-128)"))
    assert preferencias.elegir_carpeta_sistema(tmp_path) is None
