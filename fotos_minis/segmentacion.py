"""Segmentación del sujeto (figura + peana) y refinado de la máscara alfa."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

MODELOS = {
    "combinado": "Combinado BiRefNet + ISNet (recomendado)",
    "birefnet-general": "BiRefNet (máxima calidad en detalles finos, más lento)",
    "isnet-general-use": "ISNet (rápido, buena calidad)",
    "birefnet-general-lite": "BiRefNet lite (intermedio)",
    "u2net": "U²-Net (clásico, ligero)",
}
MODELO_POR_DEFECTO = "combinado"
# Modelos que forman el modo combinado. Cada uno falla en casos distintos: BiRefNet
# puede tomar una peana gris sobre tapete gris por fondo; ISNet es más tosco en el
# contorno. La unión conserva lo que cualquiera de los dos reconoce como sujeto y
# el refinado posterior elimina las motas sueltas.
_COMBINADO = ("birefnet-general", "isnet-general-use")


@lru_cache(maxsize=2)  # BiRefNet ocupa ~8 GB en CPU; ISNet ~1 GB
def _sesion(modelo: str):
    import onnxruntime as ort
    from rembg import new_session

    opciones = ort.SessionOptions()
    opciones.enable_cpu_mem_arena = False  # libera la memoria entre inferencias
    return new_session(modelo, sess_opts=opciones)


def mascara_bruta(imagen: Image.Image, modelo: str = MODELO_POR_DEFECTO) -> np.ndarray:
    """Devuelve la máscara del modelo como float32 en [0, 1] al tamaño de la imagen."""
    from rembg import remove

    if modelo == "combinado":
        return np.maximum.reduce([mascara_bruta(imagen, m) for m in _COMBINADO])

    m = remove(imagen.convert("RGB"), session=_sesion(modelo), only_mask=True,
               post_process_mask=False)
    m = np.asarray(m, dtype=np.float32) / 255.0
    if m.shape[:2] != (imagen.height, imagen.width):
        m = cv2.resize(m, (imagen.width, imagen.height), interpolation=cv2.INTER_CUBIC)
    return np.clip(m, 0.0, 1.0)


def _filtro_guiado(guia: np.ndarray, p: np.ndarray, radio: int, eps: float,
                   submuestreo: int = 1) -> np.ndarray:
    """Filtro guiado rápido (He et al.) con guía en color.

    Ajusta la transición de la máscara a los bordes reales de la imagen, lo que
    recupera detalles finos (lanzas, pelo, capas, filos de la peana). Los
    coeficientes se calculan a resolución reducida y se aplican a resolución
    completa, así que el detalle se conserva y la memoria se mantiene acotada.
    """
    h, w = p.shape
    s = max(int(submuestreo), 1)
    if s > 1:
        tam = (max(w // s, 1), max(h // s, 1))
        I = cv2.resize(guia, tam, interpolation=cv2.INTER_AREA)
        pp = cv2.resize(p, tam, interpolation=cv2.INTER_AREA)
        r_ = max(radio // s, 1)
    else:
        I, pp, r_ = guia, p, radio
    k = (2 * r_ + 1, 2 * r_ + 1)
    box = lambda x: cv2.blur(x, k, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    mI = box(I)
    mp = box(pp)
    cov_Ip = box(I * pp[..., None]) - mI * mp[..., None]
    var = np.empty(I.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = box(I[..., i] * I[..., j]) - mI[..., i] * mI[..., j]
            var[..., i, j] = var[..., j, i] = v
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov_Ip[..., None])[..., 0]
    b = mp - np.einsum("hwc,hwc->hw", a, mI)
    a, b = box(a), box(b)
    if s > 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.einsum("hwc,hwc->hw", a, guia) + b


def refinar_mascara(
    imagen: np.ndarray,
    alfa: np.ndarray,
    *,
    limpiar_islas: bool = True,
    rellenar_huecos_px: int = 0,
    refinar_bordes: bool = True,
) -> np.ndarray:
    """Limpia y afina la máscara.

    - Elimina islas sueltas del fondo (conserva el sujeto principal y las piezas
      grandes separadas, p. ej. un arma o una peana que no toque la figura).
    - Opcionalmente rellena huecos pequeños dentro del sujeto (errores del modelo).
    - Ajusta la transición del borde a los bordes reales de la imagen.
    """
    alfa = alfa.astype(np.float32)
    h, w = alfa.shape
    binaria = (alfa > 0.5).astype(np.uint8)

    if limpiar_islas and binaria.any():
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            mayor = areas.max()
            conservar = np.zeros(n, bool)
            conservar[1:] = areas >= max(0.02 * mayor, 64)
            fuera = ~conservar[etiquetas] & (etiquetas > 0)
            # Quita también la penumbra alrededor de las islas eliminadas.
            fuera = cv2.dilate(fuera.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            cerca_sujeto = cv2.dilate(
                (conservar[etiquetas] & (etiquetas > 0)).astype(np.uint8),
                np.ones((5, 5), np.uint8)) > 0
            alfa[fuera & ~cerca_sujeto] = 0.0
            binaria = (alfa > 0.5).astype(np.uint8)
        # Penumbras débiles lejos del sujeto (ruido del modelo).
        lejos = cv2.dilate(binaria, np.ones((15, 15), np.uint8)) == 0
        alfa[lejos & (alfa < 0.35)] = 0.0

    if rellenar_huecos_px > 0:
        inv = (1 - binaria).astype(np.uint8)
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        for k in range(1, n):
            x, y, ww, hh, area = stats[k]
            toca_borde = x == 0 or y == 0 or x + ww == w or y + hh == h
            if not toca_borde and area <= rellenar_huecos_px:
                alfa[etiquetas == k] = 1.0

    if refinar_bordes:
        guia = imagen.astype(np.float32) / 255.0
        escala = max(h, w) / 2048.0
        radio = int(np.clip(round(4 * max(escala, 0.5)), 2, 12))
        borde = cv2.dilate((alfa > 0.02).astype(np.uint8), np.ones((radio * 4 + 1,) * 2, np.uint8)) \
            & cv2.dilate((alfa < 0.98).astype(np.uint8), np.ones((radio * 4 + 1,) * 2, np.uint8))
        if borde.any():
            ys, xs = np.nonzero(borde)
            pad = radio * 3
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
            area = (y1 - y0) * (x1 - x0)
            sub_m = int(np.ceil(np.sqrt(area / 2.5e6)))  # ~2,5 Mpx máximo
            fino = _filtro_guiado(guia[y0:y1, x0:x1], alfa[y0:y1, x0:x1], radio, 1e-4, sub_m)
            fino = np.clip(fino, 0.0, 1.0)
            zona = borde[y0:y1, x0:x1] > 0
            sub = alfa[y0:y1, x0:x1]
            sub[zona] = fino[zona]

    # El núcleo sólido del sujeto queda exactamente a 1 (sin cambios en sus píxeles).
    nucleo = cv2.erode((alfa > 0.9).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    alfa[nucleo] = 1.0
    alfa[alfa < 0.01] = 0.0
    return np.clip(alfa, 0.0, 1.0)
