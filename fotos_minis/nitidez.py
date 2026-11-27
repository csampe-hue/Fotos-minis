"""Restauración del desenfoque por distancia (profundidad de campo).

En una foto de miniaturas con poca profundidad de campo no todo está igual de
desenfocado: la figura de delante puede estar nítida y la de detrás no, o la peana
nítida y la cabeza no. Un enfoque uniforme (o la super-resolución, que trata toda la
imagen igual) deja borroso lo que más lo está y exagera lo que ya era nítido.

Aquí se hace en dos pasos:

1. **Mapa de desenfoque.** En cada borde del sujeto se mide cuánto está desenfocado
   comparando su gradiente con el de una copia desenfocada a propósito (método de
   Zhuo y Sim, 2011): un borde nítido pierde mucho contraste al desenfocarlo y uno ya
   borroso casi nada. De esa razón sale el radio (sigma, en píxeles) del desenfoque
   gaussiano equivalente. Esas medidas sueltas se extienden a todo el sujeto con una
   media ponderada local, así que el mapa varía suavemente con la distancia.
2. **Deconvolución variable.** La imagen se restaura (filtro de Wiener con
   regularización de Laplaciana) para varios radios fijos y en cada píxel se mezclan
   los dos más cercanos a su valor del mapa. Para evitar halos, el resultado se limita
   al rango de valores del entorno.

Sólo se cambia la luminancia del sujeto (el fondo se sustituye después), y sólo si se
pide (``OpcionesEstudio.nitidez`` > 0): por defecto la figura sigue siendo bit a bit
la original.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import cv2
import numpy as np

SIGMA_REF = 1.2               # desenfoque de referencia para medir (px)
SIGMA_MAX = 6.0               # más allá la restauración no recupera nada útil
NIVELES = (0.0, 0.7, 1.1, 1.6, 2.2, 3.0, 4.0, 5.0, 6.0)   # radios restaurados (px)
LAMBDA = 0.004                # regularización (más alto = menos ruido y menos nitidez)
HOLGURA = 0.03                # cuánto puede pasarse del rango local (evita halos)
RADIO = 1.5                   # radio del entorno para ese rango, en sigmas
PROCESO_MAX = 6e6             # píxeles de la zona del sujeto procesados a escala real

_cache: OrderedDict = OrderedDict()


# --------------------------------------------------------------------------- #
# Mapa de desenfoque
# --------------------------------------------------------------------------- #

def _gradiente(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _desenfocar(x: np.ndarray, s: float) -> np.ndarray:
    """Desenfoque gaussiano ancho, hecho a escala reducida (mucho más rápido y, para
    radios grandes, prácticamente igual)."""
    f = max(1.0, s / 3.0)
    if f <= 1.0:
        return cv2.GaussianBlur(x, (0, 0), s)
    h, w = x.shape
    tam = (max(int(round(w / f)), 1), max(int(round(h / f)), 1))
    peq = cv2.GaussianBlur(cv2.resize(x, tam, interpolation=cv2.INTER_AREA), (0, 0), s / f)
    return cv2.resize(peq, (w, h), interpolation=cv2.INTER_LINEAR)


def _bordes(g: np.ndarray, zona: np.ndarray) -> np.ndarray:
    """Bordes fiables (Canny con umbrales automáticos medidos dentro del sujeto)."""
    g8 = np.clip(g * 255, 0, 255).astype(np.uint8)
    suave = cv2.GaussianBlur(g8, (0, 0), 0.8)
    mag = _gradiente(suave.astype(np.float32))
    vals = mag[zona]
    if vals.size == 0:
        return np.zeros_like(zona)
    alto = float(np.percentile(vals, 90))
    alto = max(alto, 40.0)
    return (cv2.Canny(suave, alto * 0.4, alto, L2gradient=True) > 0) & zona


def mapa_desenfoque(gris: np.ndarray, alfa: np.ndarray) -> np.ndarray:
    """Sigma del desenfoque (px) en cada píxel, a partir de los bordes del sujeto.

    ``gris`` es la luminancia en [0, 1]. Donde no hay bordes cerca se usa una media de
    más alcance, y si el sujeto no tiene bordes utilizables el mapa vale 0.
    """
    zona = alfa > 0.3
    if not zona.any():
        return np.zeros(gris.shape, np.float32)
    g = gris.astype(np.float32)
    bordes = _bordes(g, zona)
    if bordes.sum() < 20:
        return np.zeros(gris.shape, np.float32)

    g1 = _gradiente(g)
    g2 = _gradiente(cv2.GaussianBlur(g, (0, 0), SIGMA_REF))
    # El máximo del gradiente puede no caer justo en el píxel de Canny: se toma el
    # máximo en una ventana pequeña (más estable en bordes ruidosos).
    k = np.ones((3, 3), np.uint8)
    g1m = cv2.dilate(g1, k)
    g2m = cv2.dilate(g2, k)
    r = g1m[bordes] / np.maximum(g2m[bordes], 1e-6)
    sig = np.full(r.shape, SIGMA_MAX, np.float32)
    ok = r > 1.0 + 1e-3
    sig[ok] = SIGMA_REF / np.sqrt(r[ok] ** 2 - 1.0)
    # El Sobel de 3×3 ya desenfoca un poco (≈0,5 px): se descuenta.
    sig = np.sqrt(np.maximum(sig ** 2 - 0.25, 0.0))
    sig = np.clip(sig, 0.0, SIGMA_MAX)
    # Más peso a los bordes con contraste (menos afectados por el ruido).
    peso = np.clip(g1[bordes] / max(float(np.percentile(g1[bordes], 75)), 1e-6), 0.2, 1.0)

    ys, xs = np.nonzero(bordes)
    num = np.zeros(gris.shape, np.float32)
    den = np.zeros(gris.shape, np.float32)
    num[ys, xs] = sig * peso
    den[ys, xs] = peso

    ys_z, xs_z = np.nonzero(zona)
    tam = max(ys_z.max() - ys_z.min(), xs_z.max() - xs_z.min(), 1)
    # Escala de la media local: la profundidad cambia de forma suave dentro de la figura.
    s1 = max(tam / 30.0, 4.0)
    a, b = _desenfocar(num, s1), _desenfocar(den, s1)
    a2, b2 = _desenfocar(num, 4 * s1), _desenfocar(den, 4 * s1)
    glob = float((sig * peso).sum() / peso.sum())
    fino = a / np.maximum(b, 1e-12)
    grueso = (a2 + 1e-4 * glob) / (b2 + 1e-4)
    # Confianza de la media fina según la densidad de bordes cercanos.
    conf = np.clip(b / (b.max() * 0.05 + 1e-12), 0.0, 1.0)
    mapa = conf * fino + (1 - conf) * grueso
    return np.clip(mapa, 0.0, SIGMA_MAX).astype(np.float32)


# --------------------------------------------------------------------------- #
# Deconvolución
# --------------------------------------------------------------------------- #

def _wiener(f_y: np.ndarray, forma: tuple[int, int], sigma: float, lam: float) -> np.ndarray:
    """Deconvolución gaussiana de la imagen (ya en frecuencia, rfft2)."""
    h, w = forma
    fy = np.fft.fftfreq(h)[:, None].astype(np.float32)
    fx = np.fft.rfftfreq(w)[None, :].astype(np.float32)
    w2 = (2 * np.pi) ** 2 * (fx * fx + fy * fy)
    k = np.exp(-0.5 * sigma * sigma * w2)                  # PSF gaussiana
    lap = (2 * np.cos(2 * np.pi * fx) + 2 * np.cos(2 * np.pi * fy) - 4)
    filtro = k / (k * k + lam * lap * lap)
    return np.fft.irfft2(f_y * filtro, s=forma).astype(np.float32)


def _restaurar_y(y: np.ndarray, mapa: np.ndarray) -> np.ndarray:
    """Deconvolución de la luminancia con el radio de ``mapa`` en cada píxel."""
    pad = int(np.ceil(3 * SIGMA_MAX)) + 2
    yp = cv2.copyMakeBorder(y, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    f_y = np.fft.rfft2(yp)
    h, w = y.shape
    res = y.copy()
    acum = np.zeros_like(y)
    peso_tot = np.zeros_like(y)
    for i, s in enumerate(NIVELES):
        izq = NIVELES[i - 1] if i > 0 else None
        der = NIVELES[i + 1] if i + 1 < len(NIVELES) else None
        # Peso triangular: 1 en el nivel, 0 en los vecinos.
        p = np.zeros_like(y)
        if izq is not None:
            m = (mapa >= izq) & (mapa <= s)
            p[m] = (mapa[m] - izq) / (s - izq)
        if der is not None:
            m = (mapa > s) & (mapa < der)
            p[m] = (der - mapa[m]) / (der - s)
        else:
            p[mapa >= s] = 1.0
        if i == 0:
            p[mapa <= s] = 1.0
        if not (p > 1e-4).any():
            continue
        if s <= 0:
            nivel = y
        else:
            # Más regularización cuanto mayor el radio (más ruido que amplificar).
            nivel = _wiener(f_y, yp.shape, s, LAMBDA * (1 + 0.15 * s))[pad:pad + h, pad:pad + w]
            # Sin halos: no se sale del rango de valores del entorno (radio ≈ sigma).
            r = max(int(np.ceil(RADIO * s)), 1)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            lo, hi = cv2.erode(y, k), cv2.dilate(y, k)
            nivel = np.clip(nivel, lo - HOLGURA, hi + HOLGURA)
        acum += p * nivel
        peso_tot += p
    m = peso_tot > 1e-6
    res[m] = acum[m] / peso_tot[m]
    return res


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

def _luminancia(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


def _calcular(imagen: np.ndarray, alfa: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Diferencia de luminancia restaurada − original (fuerza 1) en la caja del sujeto."""
    h, w = alfa.shape
    ys, xs = np.nonzero(alfa > 0.05)
    pad = 48
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    rgb = imagen[y0:y1, x0:x1, :3].astype(np.float32) / 255.0
    a = alfa[y0:y1, x0:x1].astype(np.float32)
    y = _luminancia(rgb)
    escala = min(1.0, float(np.sqrt(PROCESO_MAX / y.size)))
    if escala < 1.0:
        # Sujeto enorme: el mapa se mide y aplica a escala reducida (el desenfoque en
        # píxeles reales es proporcionalmente mayor, así que la corrección sigue valiendo).
        tam = (max(int((x1 - x0) * escala), 1), max(int((y1 - y0) * escala), 1))
        ys_, as_ = cv2.resize(y, tam, interpolation=cv2.INTER_AREA), cv2.resize(a, tam, interpolation=cv2.INTER_AREA)
        d = _restaurar_y(ys_, mapa_desenfoque(ys_, as_)) - ys_
        delta = cv2.resize(d, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    else:
        delta = _restaurar_y(y, mapa_desenfoque(y, a)) - y
    return delta.astype(np.float32), (y0, y1, x0, x1)


def _clave(imagen: np.ndarray, alfa: np.ndarray) -> str:
    h = hashlib.sha1(np.ascontiguousarray(imagen).tobytes())
    h.update(np.ascontiguousarray(alfa, np.float32).tobytes())
    return h.hexdigest()


def diferencia(imagen: np.ndarray, alfa: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Corrección de luminancia (fuerza 1) y su caja; se calcula una vez por foto y máscara."""
    if not (alfa > 0.05).any():
        return None
    k = _clave(imagen, alfa)
    if k not in _cache:
        _cache[k] = _calcular(imagen, alfa)
        while len(_cache) > 3:
            _cache.popitem(last=False)
    _cache.move_to_end(k)
    return _cache[k]


def restaurar(imagen: np.ndarray, alfa: np.ndarray, fuerza: float = 1.0) -> np.ndarray:
    """Devuelve ``imagen`` (uint8 RGB) con el desenfoque por distancia corregido en el
    sujeto. ``fuerza`` 0-1 (0 = sin cambios). Fuera del sujeto no cambia nada."""
    fuerza = float(np.clip(fuerza, 0.0, 1.0))
    if fuerza <= 0:
        return imagen
    r = diferencia(imagen, alfa)
    if r is None:
        return imagen
    delta, (y0, y1, x0, x1) = r
    a = np.clip(alfa[y0:y1, x0:x1].astype(np.float32) * 2.0, 0.0, 1.0)   # todo el sujeto, también el borde
    sub = imagen[y0:y1, x0:x1, :3].astype(np.float32)
    sub = sub + (fuerza * 255.0 * delta * a)[..., None]
    out = imagen.copy()
    out[y0:y1, x0:x1, :3] = np.clip(np.round(sub), 0, 255).astype(np.uint8)
    return out


def mapa(imagen: np.ndarray, alfa: np.ndarray) -> np.ndarray:
    """Mapa de desenfoque (px) de la foto completa, para inspección y pruebas."""
    y = _luminancia(imagen[..., :3].astype(np.float32) / 255.0)
    return mapa_desenfoque(y, alfa.astype(np.float32))


def aplicar(imagen: np.ndarray, alfa: np.ndarray, fuerza: float,
            primer_plano: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Restaura ``imagen`` y, si se da, también el primer plano ya descontaminado
    (float 0-1, misma forma), sumándole la misma corrección."""
    if fuerza <= 0:
        return imagen, primer_plano
    nueva = restaurar(imagen, alfa, fuerza)
    if primer_plano is not None and nueva is not imagen:
        cambio = (nueva[..., :3].astype(np.float32) - imagen[..., :3].astype(np.float32)) / 255.0
        primer_plano = np.clip(primer_plano + cambio, 0.0, 1.0).astype(np.float32)
    return nueva, primer_plano
