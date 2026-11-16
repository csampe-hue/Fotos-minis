"""Composición sobre fondo negro de estudio.

Principio rector: los píxeles opacos del sujeto (figura y peana) se copian tal
cual. Sólo se recalculan los píxeles semitransparentes del borde (para eliminar
el color del fondo antiguo que se "cuela" en el contorno) y, si el usuario lo
pide explícitamente, se corrige la dominante de color de la iluminación.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OpcionesEstudio:
    # Fondo
    negro: int = 0                  # nivel del negro de fondo (0 = negro puro)
    halo: float = 0.10              # intensidad del foco suave detrás del sujeto (0-1)
    halo_tamano: float = 1.0        # radio del foco relativo a la altura del sujeto
    vineteado: float = 0.35         # oscurecimiento de las esquinas (0-1)
    # Suelo
    sombra: float = 0.6             # sombra de contacto bajo la peana (0-1)
    reflejo: float = 0.0            # reflejo en suelo negro brillante (0-1)
    reflejo_caida: float = 0.25     # longitud del reflejo relativa a la altura del sujeto
    # Horizonte: transición visible del suelo a la pared del fondo
    horizonte: float = 0.0          # intensidad (0 = sin horizonte, fondo negro continuo)
    horizonte_distancia: float = 3.0  # distancia suelo-pared detrás de la peana (en diámetros de peana)
    horizonte_desenfoque: float = 0.5  # desenfoque de la línea (0 = nítida; crece con la distancia)
    horizonte_suelo: float = 0.5    # luminosidad del suelo respecto a la pared (0-1)
    horizonte_altura: float = 0.0   # ajuste manual de la altura (fracción de la altura del sujeto, + = más arriba)
    # Integración del sujeto
    descontaminar_bordes: bool = True   # quita el color del fondo antiguo del contorno
    oscurecer_borde: float = 0.15       # "light wrap" negro: el entorno oscuro envuelve el borde
    corregir_dominante: float = 0.0     # 0 = colores intactos; 1 = neutraliza la luz por completo
    # Encuadre
    reencuadrar: bool = True
    margen: float = 0.12            # margen alrededor del sujeto (fracción del lado mayor)
    proporcion: str = "original"    # "original", "1:1", "4:5", "3:2", "2:3", "16:9"


PROPORCIONES = {"1:1": 1.0, "4:5": 4 / 5, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def _srgb_a_lineal(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lineal_a_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def caja_sujeto(alfa: np.ndarray, umbral: float = 0.05) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(alfa > umbral)
    if len(ys) == 0:
        h, w = alfa.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def geometria_peana(alfa: np.ndarray) -> dict:
    """Estima la posición y anchura de la base del sujeto (la peana).

    Se analiza la franja inferior de la silueta: su anchura da el diámetro de la
    peana y su borde inferior, la línea de contacto con el suelo.
    """
    x0, y0, x1, y1 = caja_sujeto(alfa)
    alto = y1 - y0
    franja = alfa[max(y1 - max(int(alto * 0.15), 2), y0):y1, x0:x1] > 0.5
    cols = np.nonzero(franja.any(axis=0))[0]
    if len(cols) == 0:
        cx, ancho = (x0 + x1) / 2, (x1 - x0)
    else:
        cx, ancho = x0 + (cols.min() + cols.max()) / 2, cols.max() - cols.min() + 1
    # Línea de contacto: la fila más baja del sujeto en cada columna, mediana central.
    colsub = alfa[:, x0:x1] > 0.5
    fondo_col = np.where(colsub.any(axis=0), colsub.shape[0] - 1 - np.argmax(colsub[::-1], axis=0), -1)
    validos = fondo_col[fondo_col >= 0]
    y_suelo = float(np.percentile(validos, 95)) if len(validos) else float(y1)

    # Inclinación de la cámara a partir de la peana: una peana redonda vista con un
    # ángulo θ sobre la mesa es una elipse de proporción alto/ancho = sen θ. En el
    # borde inferior de la silueta, los extremos izquierdo y derecho están a la
    # altura del centro de la elipse y el punto más bajo, un semieje menor por debajo.
    inclinacion, y_centro = 0.3, y_suelo - 0.3 * ancho / 2
    a_ = ancho / 2
    if a_ > 4:
        rel = np.arange(x0, x1) + 0.5 - cx
        u = np.abs(rel) / a_
        extremos = (u > 0.9) & (u <= 1.0) & (fondo_col >= 0)
        if extremos.sum() >= 2:
            # Cerca de los extremos el borde ya ha bajado b·sqrt(1-u²): se descuenta.
            y_ext = float(np.median(fondo_col[extremos]))
            caida = float(np.median(np.sqrt(np.clip(1 - u[extremos] ** 2, 0, 1))))
            b = (y_suelo - y_ext) / max(1.0 - caida, 0.3)
            inclinacion = float(np.clip(b / a_, 0.08, 0.8))
            y_centro = y_suelo - b
    return {"cx": float(cx), "ancho": float(ancho), "y_suelo": y_suelo,
            "caja": (x0, y0, x1, y1), "alto": float(alto),
            "inclinacion": inclinacion, "y_centro": y_centro}


# --------------------------------------------------------------------------- #
# Integración del sujeto
# --------------------------------------------------------------------------- #

def descontaminar(imagen: np.ndarray, alfa: np.ndarray) -> np.ndarray:
    """Estima el color real del sujeto en los píxeles de borde (sin el fondo viejo).

    Devuelve la imagen de primer plano en float [0,1]. Donde alfa == 1 es
    idéntica a la original.
    """
    from pymatting import estimate_foreground_ml

    img = imagen.astype(np.float64) / 255.0
    frg = img.copy()
    borde = (alfa > 0.0) & (alfa < 1.0)
    if not borde.any():
        return frg.astype(np.float32)
    ys, xs = np.nonzero(borde)
    pad = 16
    h, w = alfa.shape
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    sub_img = img[y0:y1, x0:x1]
    sub_a = alfa[y0:y1, x0:x1].astype(np.float64)
    # Para imágenes muy grandes se trabaja a escala reducida y sólo se usa en el borde.
    escala = min(1.0, np.sqrt(4e6 / sub_a.size))
    if escala < 1.0:
        tam = (max(int((x1 - x0) * escala), 1), max(int((y1 - y0) * escala), 1))
        f = estimate_foreground_ml(cv2.resize(sub_img, tam, interpolation=cv2.INTER_AREA),
                                   cv2.resize(sub_a, tam, interpolation=cv2.INTER_AREA))
        f = cv2.resize(f, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    else:
        f = estimate_foreground_ml(sub_img, sub_a)
    zona = borde[y0:y1, x0:x1]
    sub_frg = frg[y0:y1, x0:x1]
    sub_frg[zona] = np.clip(f[zona], 0.0, 1.0)
    return frg.astype(np.float32)


def corregir_luz(frg: np.ndarray, imagen: np.ndarray, alfa: np.ndarray, fuerza: float) -> np.ndarray:
    """Neutraliza la dominante de color de la iluminación original.

    La dominante se mide en el fondo antiguo (normalmente un papel o pared que
    debería ser neutro) y se compensa sobre el sujeto con la fuerza indicada.
    """
    if fuerza <= 0:
        return frg
    fondo = alfa < 0.02
    if fondo.sum() < 500:
        return frg
    lin = _srgb_a_lineal(imagen[fondo].astype(np.float32) / 255.0)
    # Sólo píxeles de fondo razonablemente iluminados y no saturados.
    lum = lin.mean(axis=1)
    sel = (lum > np.percentile(lum, 30)) & (lin.max(axis=1) < 0.98)
    if sel.sum() < 100:
        sel = np.ones(len(lin), bool)
    media = lin[sel].mean(axis=0) + 1e-6
    ganancia = media.mean() / media
    ganancia = 1.0 + (ganancia - 1.0) * float(np.clip(fuerza, 0, 1))
    return _lineal_a_srgb(_srgb_a_lineal(frg) * ganancia).astype(np.float32)


# --------------------------------------------------------------------------- #
# Fondo de estudio
# --------------------------------------------------------------------------- #

def linea_horizonte(geo: dict, op: OpcionesEstudio) -> tuple[float, float]:
    """Altura (y) de la unión suelo-pared y su desenfoque (sigma, en píxeles).

    Un punto del suelo a una distancia d detrás del centro de la peana aparece
    d·sen θ más arriba en la foto (θ = inclinación de la cámara), con un ligero
    escorzo por la perspectiva. El desenfoque crece con esa distancia, porque la
    figura está enfocada y la pared queda detrás.
    """
    k = max(op.horizonte_distancia, 0.0)
    profundidad = geo["ancho"] * k / (1.0 + 0.12 * k)  # escorzo
    y = geo["y_centro"] - profundidad * geo["inclinacion"] - op.horizonte_altura * geo["alto"]
    sigma = 1.0 + op.horizonte_desenfoque * 0.01 * geo["ancho"] * k
    return float(y), float(sigma)


def fondo_estudio(h: int, w: int, geo: dict, op: OpcionesEstudio) -> np.ndarray:
    """Fondo negro de estudio (lineal): negro base + foco suave + suelo + viñeta."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    negro = _srgb_a_lineal(np.float32(op.negro / 255.0))
    fondo = np.full((h, w), negro, np.float32)

    alto = max(geo["alto"], 1.0)
    x0, y0, x1, y1 = geo["caja"]
    cy = y0 + 0.45 * (y1 - y0)
    r = alto * 0.75 * op.halo_tamano
    foco = np.exp(-(((xx - geo["cx"]) / (r * 1.15)) ** 2 + ((yy - cy) / r) ** 2) * 1.6)

    if op.horizonte > 0:
        # Pared (arriba) y suelo (abajo) separados por la línea de horizonte.
        y_h, sigma = linea_horizonte(geo, op)
        suelo_m = 0.5 * (1.0 + np.tanh((yy - y_h) / (1.2 * sigma)))
        # Charco de luz sobre el suelo, en perspectiva: un círculo en la mesa es una
        # elipse con la misma proporción que la peana.
        R = 1.8 * geo["ancho"]
        d2 = ((xx - geo["cx"]) / R) ** 2 + ((yy - geo["y_centro"]) / (R * geo["inclinacion"])) ** 2
        luz_suelo = (0.004 + 0.035 * np.exp(-d2 * 1.2)) * (0.3 + 1.4 * op.horizonte_suelo)
        luz_pared = 0.06 * op.halo * foco
        # La luz de la pared se suaviza más cuanto más lejos está (desenfoque).
        if sigma > 1.5:
            luz_pared = cv2.GaussianBlur(luz_pared, (0, 0), sigma * 0.5)
        fondo += op.horizonte * luz_suelo * suelo_m + luz_pared * (1 - suelo_m) \
            + (1 - op.horizonte) * 0.06 * op.halo * foco * suelo_m
    elif op.halo > 0:
        # El foco se apaga al llegar al suelo (perspectiva: el telón cae detrás de la peana).
        caida = np.clip((geo["y_suelo"] - yy) / (0.08 * alto) + 0.5, 0, 1)
        suelo = np.exp(-(((xx - geo["cx"]) / (r * 1.4)) ** 2 + ((yy - geo["y_suelo"]) / (r * 0.25)) ** 2))
        fondo += 0.06 * op.halo * (foco * (0.55 + 0.45 * caida) + 0.5 * suelo)

    if op.vineteado > 0:
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        fondo *= 1.0 - op.vineteado * np.clip(d, 0, 1) ** 2

    if op.sombra > 0:
        # Sombra de contacto: elipse en perspectiva bajo la peana.
        rx = geo["ancho"] * 0.55
        ry = max(geo["ancho"] * 0.09, 2.0)
        d2 = ((xx - geo["cx"]) / rx) ** 2 + ((yy - geo["y_suelo"]) / ry) ** 2
        s = np.exp(-d2 * 2.0)
        fondo *= 1.0 - op.sombra * s
    return np.clip(fondo, 0, 1)[..., None].repeat(3, axis=2)


# --------------------------------------------------------------------------- #
# Encuadre
# --------------------------------------------------------------------------- #

def calcular_lienzo(alfa: np.ndarray, op: OpcionesEstudio) -> tuple[int, int, int, int]:
    """Devuelve (ox, oy, W, H): desplazamiento del original dentro del nuevo lienzo
    y tamaño de éste. El sujeto no se reescala nunca: sólo se recorta o amplía el lienzo."""
    h, w = alfa.shape
    if not op.reencuadrar:
        return 0, 0, w, h
    # Caja de la parte sólida del sujeto: las penumbras débiles del contorno (p. ej.
    # restos de sombra) no deben desplazar el centrado.
    x0, y0, x1, y1 = caja_sujeto(alfa, umbral=0.5)
    bw, bh = x1 - x0, y1 - y0
    extra_inf = op.reflejo_caida * bh if op.reflejo > 0 else 0
    m = int(op.margen * max(bw, bh))
    # Lienzo simétrico: el sujeto queda centrado en horizontal y en vertical. Si hay
    # reflejo, el margen inferior (y por simetría el superior) crece para que quepa.
    W, H = bw + 2 * m, int(bh + 2 * (m + extra_inf))
    prop = PROPORCIONES.get(op.proporcion)
    if prop is None:
        prop = w / h
    if W / H < prop:
        W = int(round(H * prop))
    else:
        H = int(round(W / prop))
    W, H = W + W % 2, H + H % 2
    ox = int(round((W - bw) / 2)) - x0
    oy = int(round((H - bh) / 2)) - y0
    return ox, oy, W, H


def _pegar(dst: np.ndarray, src: np.ndarray, ox: int, oy: int) -> None:
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    xa, ya = max(ox, 0), max(oy, 0)
    xb, yb = min(ox + w, W), min(oy + h, H)
    if xa >= xb or ya >= yb:
        return
    dst[ya:yb, xa:xb] = src[ya - oy:yb - oy, xa - ox:xb - ox]


# --------------------------------------------------------------------------- #
# Composición final
# --------------------------------------------------------------------------- #

def componer(imagen: np.ndarray, alfa: np.ndarray, op: OpcionesEstudio) -> np.ndarray:
    """Compone el sujeto sobre el fondo de estudio. Devuelve uint8 RGB."""
    alfa = alfa.astype(np.float32)
    frg = descontaminar(imagen, alfa) if op.descontaminar_bordes \
        else imagen.astype(np.float32) / 255.0
    frg = corregir_luz(frg, imagen, alfa, op.corregir_dominante)

    ox, oy, W, H = calcular_lienzo(alfa, op)
    a_l = np.zeros((H, W), np.float32)
    f_l = np.zeros((H, W, 3), np.float32)
    _pegar(a_l, alfa, ox, oy)
    _pegar(f_l, frg, ox, oy)

    geo = geometria_peana(a_l)
    fondo = fondo_estudio(H, W, geo, op)  # lineal

    if op.reflejo > 0:
        # Reflejo especular respecto a la línea de contacto de la peana.
        ys = int(round(geo["y_suelo"]))
        largo = max(int(op.reflejo_caida * geo["alto"]), 1)
        a_r = np.zeros_like(a_l)
        f_r = np.zeros_like(f_l)
        n = min(ys + 1, H - ys - 1)
        if n > 0:
            a_r[ys + 1:ys + 1 + n] = a_l[ys - n + 1:ys + 1][::-1]
            f_r[ys + 1:ys + 1 + n] = f_l[ys - n + 1:ys + 1][::-1]
            dist = np.arange(H, dtype=np.float32) - ys
            atten = np.clip(1.0 - dist / largo, 0, 1) ** 1.5 * (dist > 0)
            a_r *= atten[:, None] * op.reflejo
            a_r = cv2.GaussianBlur(a_r, (0, 0), max(geo["alto"] / 400, 0.6))
            f_r_lin = _srgb_a_lineal(f_r)
            fondo = fondo * (1 - a_r[..., None]) + f_r_lin * a_r[..., None]

    # Mezcla en espacio lineal (físicamente correcta en los bordes).
    f_lin = _srgb_a_lineal(f_l)
    if op.oscurecer_borde > 0:
        # El entorno negro "envuelve" ligeramente el contorno, como en una foto real.
        r = max(geo["alto"] / 250, 1.0)
        interior = cv2.GaussianBlur(a_l, (0, 0), r)
        envolver = np.clip((1 - interior) * a_l * 2.0, 0, 1) * op.oscurecer_borde
        f_lin = f_lin * (1 - envolver[..., None])
    a3 = a_l[..., None]
    salida = _lineal_a_srgb(f_lin * a3 + fondo * (1 - a3))
    out = np.round(salida * 255).astype(np.uint8)

    # Garantía de fidelidad: el interior opaco del sujeto es bit a bit el original
    # (salvo que se haya pedido corregir la dominante de color).
    if op.corregir_dominante <= 0:
        orig = np.zeros((H, W, 3), np.uint8)
        _pegar(orig, imagen, ox, oy)
        nucleo = a_l >= 1.0
        if op.oscurecer_borde > 0:
            nucleo &= envolver < 1e-3
        out[nucleo] = orig[nucleo]
    return out


def recorte_transparente(imagen: np.ndarray, alfa: np.ndarray, descontaminar_bordes: bool = True) -> np.ndarray:
    """PNG RGBA del sujeto recortado (útil para otros montajes)."""
    frg = descontaminar(imagen, alfa) if descontaminar_bordes else imagen / 255.0
    rgba = np.dstack([np.round(np.clip(frg, 0, 1) * 255), np.round(alfa * 255)]).astype(np.uint8)
    return rgba
