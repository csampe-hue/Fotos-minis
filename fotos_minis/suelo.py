"""Suelo: detección del suelo real de la foto y suelos con textura generada.

- ``detectar_suelo``: separa en la foto original el suelo/tapete sobre el que está
  la miniatura de la pared del fondo, para poder conservarlo y sustituir sólo la pared.
- ``suelo_en_perspectiva``: genera un suelo con textura (oscura o árida) proyectado
  con la misma perspectiva que la peana y con desenfoque creciente con la distancia.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from scipy import fft as sfft

TEXTURAS = {"oscuro": "Textura oscura", "arido": "Terreno árido"}


# --------------------------------------------------------------------------- #
# Suelo real de la foto
# --------------------------------------------------------------------------- #

def detectar_suelo(imagen: np.ndarray, alfa: np.ndarray) -> np.ndarray | None:
    """Máscara (float 0-1) del suelo de la foto, o None si no se distingue.

    La unión suelo-pared es una línea larga que cruza la foto de lado a lado. Se
    trabaja a escala reducida, con la figura borrada (inpainting):
      1. Para cada píxel se mide cuánto cambian color y textura entre una ventana
         justo encima y otra justo debajo (las dos superficies difieren).
      2. Programación dinámica: se busca el camino continuo de izquierda a derecha,
         con pendiente limitada, que maximiza ese cambio (como un «seam carving»).
      3. La línea se alisa y se convierte en máscara a la resolución original.
    """
    h, w = alfa.shape
    s = min(1.0, 400.0 / max(h, w))
    tam = (max(int(w * s), 16), max(int(h * s), 16))
    img = cv2.resize(imagen, tam, interpolation=cv2.INTER_AREA)
    a = cv2.resize(alfa.astype(np.float32), tam, interpolation=cv2.INTER_AREA)
    hh, ww = a.shape
    sujeto = cv2.dilate((a > 0.2).astype(np.uint8), np.ones((9, 9), np.uint8))
    if sujeto.sum() == 0:
        return None
    limpio = cv2.inpaint(img, sujeto, 9, cv2.INPAINT_TELEA)

    # Rasgos: color (Lab) y textura (energía local de detalle de la luminosidad).
    lab = cv2.cvtColor(limpio, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    detalle = np.abs(L - cv2.GaussianBlur(L, (0, 0), 2.0))
    textura_ = cv2.GaussianBlur(detalle, (0, 0), 2.0)
    rasgos = np.dstack([lab[..., 0], lab[..., 1] * 1.5, lab[..., 2] * 1.5, textura_ * 3.0])
    # Promedio horizontal amplio: la línea es casi horizontal y así se ignora el grano.
    rasgos = cv2.GaussianBlur(rasgos, (0, 0), sigmaX=max(ww * 0.012, 1.5), sigmaY=0.8)

    # Diferencia entre la ventana de arriba y la de abajo de cada fila.
    v = max(int(hh * 0.035), 3)
    acum = np.cumsum(np.pad(rasgos, ((1, 0), (0, 0), (0, 0))), axis=0)
    fuerza = np.zeros((hh, ww), np.float32)
    ys_ = np.arange(v, hh - v)
    arriba = (acum[ys_] - acum[ys_ - v]) / v
    abajo = (acum[ys_ + v] - acum[ys_]) / v
    fuerza[ys_] = np.linalg.norm(arriba - abajo, axis=2)

    # Zona de búsqueda: por encima de la mitad de la peana; la figura no aporta.
    ys, xs = np.nonzero(sujeto)
    y_lim = int(np.percentile(ys, 85))
    fuerza[y_lim:] = 0
    fuerza[: max(int(hh * 0.03), v)] = 0
    fuerza[sujeto > 0] = 0
    if fuerza.max() <= 0:
        return None

    # Programación dinámica: camino de izquierda a derecha con |dy| <= 1 por columna.
    coste = fuerza.copy()
    origen = np.zeros((hh, ww), np.int8)
    for x in range(1, ww):
        prev = coste[:, x - 1]
        opciones = np.stack([np.r_[prev[1:], -1e9], prev, np.r_[-1e9, prev[:-1]]])  # de y+1, y, y-1
        mejor = np.argmax(opciones, axis=0)
        coste[:, x] += opciones[mejor, np.arange(hh)]
        origen[:, x] = mejor - 1
    camino = np.zeros(ww, np.float32)
    y = int(np.argmax(coste[:, -1]))
    for x in range(ww - 1, -1, -1):
        camino[x] = y
        y = int(np.clip(y - int(origen[y, x]), 0, hh - 1))  # origen: -1 venía de y+1, +1 de y-1

    # ¿Es un borde real? El cambio medio a lo largo del camino (fuera de la figura)
    # debe destacar sobre el típico de la foto.
    # Cerca de la figura el camino no es fiable (sigue el borde de la zona borrada):
    # allí se interpola en línea recta entre los tramos fiables de ambos lados.
    cerca = cv2.dilate(sujeto, np.ones((max(int(hh * 0.05), 3),) * 2, np.uint8))
    libres = cerca[camino.astype(int), np.arange(ww)] == 0
    if libres.sum() < ww * 0.25:
        return None
    a_lo_largo = fuerza[camino.astype(int), np.arange(ww)][libres]
    if np.median(a_lo_largo) < 2.5 * np.median(fuerza[fuerza > 0]):
        return None
    xs_ = np.arange(ww)
    camino = np.interp(xs_, xs_[libres], camino[libres]).astype(np.float32)
    camino = cv2.GaussianBlur(camino.reshape(1, -1), (0, 0), sigmaX=max(ww * 0.01, 1.0)).ravel()

    tope_orig = cv2.resize(camino.reshape(1, -1), (w, 1), interpolation=cv2.INTER_LINEAR).ravel() / s
    yy = np.arange(h, dtype=np.float32)[:, None]
    transicion = max(1.0, 0.0015 * max(h, w))
    return np.clip((yy - tope_orig[None, :]) / (2 * transicion) + 0.5, 0, 1).astype(np.float32)


def _mediana_movil(v: np.ndarray, k: int) -> np.ndarray:
    r = k // 2
    p = np.pad(v, r, mode="edge")
    return np.array([np.median(p[i:i + k]) for i in range(len(v))], np.float32)


# --------------------------------------------------------------------------- #
# Texturas generadas
# --------------------------------------------------------------------------- #

TAM = 2048


def _frecuencias() -> tuple[np.ndarray, np.ndarray]:
    fy = sfft.fftfreq(TAM).astype(np.float32)[:, None]
    fx = sfft.rfftfreq(TAM).astype(np.float32)[None, :]
    return fx, fy


def _aplicar(x: np.ndarray, filtro: np.ndarray) -> np.ndarray:
    """Filtra en frecuencia (periódico: sin costuras al repetir), en 32 bits."""
    return sfft.irfft2(sfft.rfft2(x, workers=-1) * filtro, s=x.shape, workers=-1).astype(np.float32)


def _normalizar(r: np.ndarray) -> np.ndarray:
    return ((r - r.mean()) / (r.std() + 1e-9)).astype(np.float32)


def _blanco(rng) -> np.ndarray:
    return rng.standard_normal((TAM, TAM), dtype=np.float32)


def _filtrar(rng, beta: float, corte: float | None = None) -> np.ndarray:
    """Ruido periódico con espectro 1/f^beta."""
    fx, fy = _frecuencias()
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    filtro = f ** (-beta / 2)
    if corte:
        filtro *= np.exp(-(f / corte) ** 2)
    filtro[0, 0] = 0.0
    return _normalizar(_aplicar(_blanco(rng), filtro))


def _gauss(sigma: float) -> np.ndarray:
    fx, fy = _frecuencias()
    return np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2)).astype(np.float32)


def _desenfocar_periodico(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    g = _gauss(sigma)
    if x.ndim == 2:
        return _aplicar(x, g)
    return np.stack([_aplicar(x[..., c], g) for c in range(x.shape[2])], axis=2)


def _alargado(rng, sx: float, sy: float) -> np.ndarray:
    """Ruido periódico de trazos alargados en horizontal (fibras)."""
    fx, fy = _frecuencias()
    g = np.exp(-2 * np.pi ** 2 * ((fx * sx) ** 2 + (fy * sy) ** 2)).astype(np.float32)
    return _normalizar(_aplicar(_blanco(rng), g))


@lru_cache(maxsize=4)
def textura(tipo: str) -> np.ndarray:
    """Textura periódica TAM×TAM en sRGB (0-1)."""
    rng = np.random.default_rng(7 if tipo == "oscuro" else 11)
    if tipo == "oscuro":
        # Superficie gris oscura de grano fino, con brillos sueltos (como un tapete de
        # neopreno o asfalto fino).
        v = (0.27 + 0.020 * _filtrar(rng, 3.0) + 0.030 * _filtrar(rng, 1.2, 0.25)
             + 0.045 * _desenfocar_periodico(_blanco(rng), 0.6))
        brillos = (rng.random((TAM, TAM)) < 0.006).astype(np.float32)
        v += 0.9 * _desenfocar_periodico(brillos, 0.7)
        rgb = v[..., None] * np.array([1.0, 1.0, 0.97], np.float32)
    elif tipo == "arido":
        # Tierra seca (como un tapete de fieltro de desierto): beige rosado claro, con
        # fibras alargadas, manchas medianas más oscuras y algún filamento claro.
        base = np.array([0.80, 0.58, 0.58], np.float32)
        manchas = _filtrar(rng, 2.4, 0.06)
        fibras = _alargado(rng, 2.2, 0.6)
        grano = _desenfocar_periodico(_blanco(rng), 0.6)
        claros = np.clip(_alargado(rng, 1.8, 0.5) - 2.2, 0, None)
        luz = 1.0 + 0.09 * manchas + 0.09 * fibras + 0.07 * grano + 0.30 * claros
        rgb = base[None, None, :] * luz[..., None]
        rgb[..., 0] += 0.02 * manchas  # zonas algo más rojizas
    else:
        raise ValueError(f"Textura desconocida: {tipo}")
    return np.clip(rgb, 0, 1).astype(np.float32)


@lru_cache(maxsize=4)
def _niveles(tipo: str) -> tuple:
    """Versiones cada vez más desenfocadas (para lejanía y desenfoque)."""
    t = textura(tipo)
    sigmas = (0.0, 1.0, 2.5, 6.0, 14.0, 32.0)
    # La transformada de cada canal se calcula una vez para todos los niveles, y los
    # niveles se guardan en 8 bits (6 niveles de 2048² ocupan ~75 MB).
    espectros = [sfft.rfft2(t[..., c], workers=-1) for c in range(3)]
    niveles = []
    for sg in sigmas:
        g = _gauss(sg)
        nivel = np.stack([sfft.irfft2(e * g, s=(TAM, TAM), workers=-1) for e in espectros], axis=2)
        niveles.append(np.round(np.clip(nivel, 0, 1) * 255).astype(np.uint8))
    return sigmas, tuple(niveles)


def precargar() -> None:
    """Genera las texturas por adelantado (se llama en segundo plano al abrir la app)."""
    for tipo in TEXTURAS:
        _niveles(tipo)


def suelo_en_perspectiva(H: int, W: int, geo: dict, tipo: str, desenfoque: float) -> np.ndarray:
    """Suelo texturizado (sRGB 0-1, H×W×3) proyectado en la perspectiva de la peana.

    Modelo proyectivo coherente con ``linea_horizonte``: la peana fija la escala
    (radio a) y la inclinación de cámara (r); las filas más altas están más lejos,
    con un punto de fuga en y_inf. Cada píxel toma la textura desenfocada según su
    tamaño en el suelo (evita el "moiré" a lo lejos) y según la profundidad de campo.
    """
    a = max(geo["ancho"] / 2, 4.0)
    r = max(geo["inclinacion"], 0.08)
    D = 8.33 * r * geo["ancho"]
    y_inf = geo["y_centro"] - D
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.maximum(yy - y_inf, 0.02 * D)
    f = D / t                                   # factor de lejanía (1 en la peana)
    u = (xx - geo["cx"]) * f / a                # coordenadas en el suelo, en radios de peana
    v = D * (f - 1) / (a * r)
    T = 220.0                                   # píxeles de textura por radio de peana
    tu = np.mod(u * T, TAM).astype(np.float32)
    tv = np.mod(-v * T, TAM).astype(np.float32)

    # Tamaño de un píxel de pantalla en la textura y desenfoque deseado (en px de textura).
    huella = T * np.sqrt((f / a) * (f * f / (a * r)))
    dof = desenfoque * 0.02 * T * np.maximum(f - 1.0, 0.0)
    sigma = np.sqrt((0.5 * huella) ** 2 + dof ** 2)

    sigmas, niveles = _niveles(tipo)
    sigmas = np.asarray(sigmas, np.float32)
    idx = np.interp(sigma, sigmas, np.arange(len(sigmas), dtype=np.float32))
    out = np.zeros((H, W, 3), np.float32)
    for i, nivel in enumerate(niveles):
        peso = np.clip(1.0 - np.abs(idx - i), 0.0, 1.0)
        if not peso.any():
            continue
        muestra = cv2.remap(nivel, tu, tv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out += muestra.astype(np.float32) * (peso[..., None] / 255.0)
    return out
