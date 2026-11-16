"""Pruebas de los perfiles (en un fichero temporal, sin tocar los del usuario)."""

import json

import pytest

from fotos_minis import perfiles


@pytest.fixture(autouse=True)
def fichero_temporal(tmp_path, monkeypatch):
    monkeypatch.setenv("FOTOS_MINIS_PERFILES", str(tmp_path / "perfiles.json"))


def test_sin_perfiles_se_usan_valores_de_fabrica():
    assert perfiles.listar() == []
    assert perfiles.predeterminado() == perfiles.FABRICA
    assert perfiles.obtener(None) == perfiles.valores_fabrica()


def test_guardar_cargar_y_predeterminado():
    v = perfiles.valores_fabrica() | {"horizonte": 0.8, "proporcion": "1:1", "modelo": "isnet-general-use"}
    perfiles.guardar("  Arena   con horizonte ", v, predeterminado=True)
    assert perfiles.listar() == ["Arena con horizonte"]
    assert perfiles.predeterminado() == "Arena con horizonte"
    assert perfiles.obtener("Arena con horizonte") == v
    op, modelo, huecos = perfiles.a_opciones(perfiles.obtener("Arena con horizonte"))
    assert op.horizonte == 0.8 and op.proporcion == "1:1" and modelo == "isnet-general-use"


def test_borrar_el_predeterminado_vuelve_a_fabrica():
    perfiles.guardar("A", perfiles.valores_fabrica(), predeterminado=True)
    perfiles.borrar("A")
    assert perfiles.listar() == [] and perfiles.predeterminado() == perfiles.FABRICA


def test_perfil_antiguo_o_incompleto_se_completa():
    perfiles.ruta().write_text(json.dumps(
        {"predeterminado": "viejo", "perfiles": {"viejo": {"halo": 0.5, "desconocido": 1, "modelo": "x"}}}))
    v = perfiles.obtener("viejo")
    assert v["halo"] == 0.5 and "desconocido" not in v
    assert v["modelo"] == perfiles.valores_fabrica()["modelo"]   # modelo no válido -> fábrica
    assert v["horizonte"] == perfiles.valores_fabrica()["horizonte"]


def test_fichero_danado_no_rompe_la_app():
    perfiles.ruta().write_text("{esto no es json")
    assert perfiles.listar() == []
    assert perfiles.ruta().with_suffix(".json.dañado").exists()


def test_nombres_no_validos():
    with pytest.raises(ValueError):
        perfiles.guardar("   ", perfiles.valores_fabrica())
    with pytest.raises(ValueError):
        perfiles.guardar(perfiles.FABRICA, perfiles.valores_fabrica())


def test_controles_de_la_interfaz_coinciden_con_el_perfil():
    """Los valores iniciales de la interfaz deben ser los de fábrica, en el mismo orden."""
    app = pytest.importorskip("app")
    assert len(app.controles) == len(perfiles.CAMPOS)
    fabrica = perfiles.valores_fabrica()
    for campo, control in zip(perfiles.CAMPOS, app.controles):
        assert control.value == fabrica[campo], campo


def test_apariencia_se_guarda_y_por_defecto_es_sistema():
    from fotos_minis import preferencias

    assert preferencias.apariencia() == "sistema"
    preferencias.fijar_apariencia("nocturno")
    assert preferencias.apariencia() == "nocturno"
    with pytest.raises(ValueError):
        preferencias.fijar_apariencia("rosa")
