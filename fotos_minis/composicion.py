"""Composición sobre fondo negro de estudio.

Principio rector: los píxeles opacos del sujeto (figura y peana) se copian tal
cual. Sólo se recalculan los píxeles semitransparentes del borde (para eliminar
el color del fondo antiguo que se "cuela" en el contorno) y, si el usuario lo
pide explícitamente, se corrige la dominante de color de la iluminación.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .suelo import TEXTURAS, detectar_suelo, suelo_en_perspectiva


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
    # Suelo: "estudio" (generado), "original" (el de la foto), "oscuro" o "arido" (texturas)
    suelo: str = "estudio"
    suelo_ajuste: float = 0.0       # sólo "original": sube (+) o baja (-) la línea suelo-pared
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
    individuales: bool = False      # además, una foto recortada por figura (si hay varias)
    superres: int = 0               # super-resolución (IA) de las fotos individuales: 0, 2 o 4
    carrusel: int = 0               # 2 a 6: imagen N:1 para dividir en N cuadrados (Instagram);
                                    # -1 = automático (el N con el que las minis ocupan más área)


SUELOS = {
    "estudio": "Estudio (generado)",
    "original": "Conservar el suelo de la foto",
    "oscuro": "Textura oscura",
    "arido": "Terreno árido",
}

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

    fondo = fondo[..., None].repeat(3, axis=2)

    if op.suelo in TEXTURAS:
        # Suelo con textura en perspectiva bajo la línea de horizonte; pared de estudio encima.
        y_h, sigma = linea_horizonte(geo, op)
        suelo_m = (0.5 * (1.0 + np.tanh((yy - y_h) / (1.2 * sigma))))[..., None]
        tex = _srgb_a_lineal(suelo_en_perspectiva(h, w, geo, op.suelo, op.horizonte_desenfoque))
        R = 1.8 * geo["ancho"]
        d2 = ((xx - geo["cx"]) / R) ** 2 + ((yy - geo["y_centro"]) / (R * geo["inclinacion"])) ** 2
        luz = (0.40 + 0.60 * np.exp(-d2 * 0.8)) * (0.4 + 1.2 * op.horizonte_suelo)
        pared = negro + 0.06 * op.halo * foco
        if sigma > 1.5:
            pared = cv2.GaussianBlur(pared, (0, 0), sigma * 0.5)
        fondo = tex * luz[..., None] * suelo_m + pared[..., None] * (1 - suelo_m)

    if op.vineteado > 0:
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        fondo *= (1.0 - op.vineteado * np.clip(d, 0, 1) ** 2)[..., None]

    if op.sombra > 0:
        fondo *= sombras_contacto(h, w, geo, op.sombra)[..., None]
    return np.clip(fondo, 0, 1)


def peanas(alfa: np.ndarray) -> list[dict]:
    """Geometría de cada peana: una por figura separada (fotos de grupo)."""
    binaria = (alfa > 0.5).astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    if n <= 1:
        return [geometria_peana(alfa)]
    mayor = stats[1:, cv2.CC_STAT_AREA].max()
    res = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 0.05 * mayor:
            continue
        res.append(geometria_peana(np.where(etiquetas == k, alfa, 0).astype(np.float32)))
    return res


def mascara_peanas(alfa: np.ndarray, lista: list[dict]) -> tuple[np.ndarray, float, float]:
    """Máscara con sólo las peanas de la silueta (sin piernas, brazos ni armas).

    Devuelve (máscara, ancho típico de peana, semieje vertical típico) en píxeles.
    """
    h, w = alfa.shape
    sujeto = alfa > 0.5
    ancho = float(np.median([g["ancho"] for g in lista]))
    r = float(np.median([g["inclinacion"] for g in lista]))
    vertical = max(ancho / 2 * r, 3.0)            # semieje vertical típico de una peana (px)

    # Núcleo de 0,6 anchos de peana: más ancho que cualquier torso, más estrecho que la peana.
    kw = max(int(ancho * 0.6), 5)
    kh = max(int(vertical * 0.25), 3)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw | 1, kh | 1))
    abierta = cv2.morphologyEx(sujeto.astype(np.uint8), cv2.MORPH_OPEN, nucleo)
    # La apertura recorta las filas más bajas de cada peana (el fondo de la elipse es
    # más estrecho que el núcleo): se recuperan dilatando dentro de la silueta.
    recuperar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw | 1, (3 * kh) | 1))
    peana = sujeto & (cv2.dilate(abierta, recuperar) > 0)
    # Y sólo la parte baja de cada silueta: a menos de 2,5 semiejes del borde inferior.
    hasta_abajo = np.full((h, w), np.inf, np.float32)
    actual = np.full(w, np.inf, np.float32)
    for y in range(h - 1, -1, -1):
        actual = np.where(sujeto[y], actual + 1, 0.0)
        hasta_abajo[y] = actual
    peana &= hasta_abajo <= 2.5 * vertical
    return peana, ancho, vertical


def sombras_contacto(h: int, w: int, geo: dict, fuerza: float) -> np.ndarray:
    """Factor multiplicativo (≤1) con la sombra de contacto de las peanas.

    Funciona con cualquier número de figuras, aunque se solapen (fotos de grupo):
    1. Las peanas se aíslan de la silueta con una apertura morfológica ancha y baja,
       que conserva las formas anchas y planas y elimina piernas, brazos y armas.
    2. Línea de contacto: justo debajo de cada borde inferior visible de una peana
       (donde la peana toca el suelo apenas llega luz) el suelo se oscurece mucho.
    3. Penumbra: un oscurecimiento amplio y suave alrededor de las peanas.
    Las distancias se miden en vertical en la escala de la peana (a·r), así que la
    sombra respeta la perspectiva del suelo.
    """
    alfa = geo.get("alfa")
    if alfa is None:
        return np.ones((h, w), np.float32)
    sujeto = alfa > 0.5
    peana, ancho, vertical = mascara_peanas(alfa, geo.get("peanas") or [geo])

    # Distancia (en filas) bajo el último borde inferior de peana, por columnas.
    borde = peana & ~np.vstack([sujeto[1:], np.zeros((1, w), bool)])
    dist = np.full((h, w), np.inf, np.float32)
    actual = np.full(w, np.inf, np.float32)
    for y in range(h):
        actual = np.where(sujeto[y], np.inf, actual + 1)
        actual = np.where(borde[y], 0.0, actual)
        dist[y] = actual
    contacto = np.exp(-(dist / max(0.18 * vertical, 1.5)) ** 2)
    contacto = cv2.GaussianBlur(contacto, (0, 0), max(vertical * 0.03, 0.7))

    penumbra = cv2.GaussianBlur(peana.astype(np.float32), (0, 0),
                                sigmaX=max(ancho * 0.10, 1.0), sigmaY=max(vertical * 0.35, 1.0))
    s = fuerza * (1.3 * contacto + 1.1 * penumbra)
    s[sujeto] = 0.0
    return (1.0 - np.clip(s, 0, 0.92)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Encuadre
# --------------------------------------------------------------------------- #

CARRUSEL_AUTO = -1


def partes_carrusel(alfa: np.ndarray, op: OpcionesEstudio) -> int:
    """Número de partes del carrusel (0 = sin carrusel). En automático se elige el
    N de 2 a 6 cuyo lienzo N:1 deja menos espacio vacío alrededor de las minis."""
    from .carrusel import PARTES

    if op.carrusel in PARTES:
        return op.carrusel
    if op.carrusel != CARRUSEL_AUTO:
        return 0
    from .carrusel import partes_recomendadas

    x0, y0, x1, y1 = caja_sujeto(alfa, umbral=0.5)
    alto = y1 - y0
    if op.reflejo > 0:
        alto += 2 * op.reflejo_caida * alto   # el lienzo reserva sitio para el reflejo
    return partes_recomendadas(x1 - x0, alto, op.margen)


def calcular_lienzo(alfa: np.ndarray, op: OpcionesEstudio,
                    dentro_de_la_foto: bool = False) -> tuple[int, int, int, int]:
    """Devuelve (ox, oy, W, H): desplazamiento del original dentro del nuevo lienzo
    y tamaño de éste. El sujeto no se reescala nunca: sólo se recorta o amplía el lienzo.

    Con ``dentro_de_la_foto`` (se conserva el suelo real) el lienzo no sale de la
    foto: así el suelo no termina en seco contra el negro.
    """
    h, w = alfa.shape
    n = partes_carrusel(alfa, op)
    if not op.reencuadrar and not n:
        return 0, 0, w, h
    # Caja de la parte sólida del sujeto: las penumbras débiles del contorno (p. ej.
    # restos de sombra) no deben desplazar el centrado.
    x0, y0, x1, y1 = caja_sujeto(alfa, umbral=0.5)
    bw, bh = x1 - x0, y1 - y0
    extra_inf = op.reflejo_caida * bh if op.reflejo > 0 else 0
    # En carrusel el margen se mide sobre el alto del grupo: con una fila larga de minis,
    # medirlo sobre el ancho daría márgenes enormes arriba y abajo y las minis quedarían
    # pequeñas en cada parte.
    m = int(op.margen * (bh if n else max(bw, bh)))
    # Lienzo simétrico: el sujeto queda centrado en horizontal y en vertical. Si hay
    # reflejo, el margen inferior (y por simetría el superior) crece para que quepa.
    W, H = bw + 2 * m, int(bh + 2 * (m + extra_inf))
    # Carrusel: proporción N:1 exacta (N cuadrados seguidos), lo más ajustada posible
    # al sujeto con el margen indicado.
    prop = float(n) if n else PROPORCIONES.get(op.proporcion)
    if prop is None:
        prop = w / h
    if W / H < prop:
        W = int(round(H * prop))
    else:
        H = int(round(W / prop))
    if dentro_de_la_foto:
        escala = min(1.0, w / W, h / H)
        W, H = max(int(W * escala), 2), max(int(H * escala), 2)
        if n:
            H = max(min(H, h, w // n), 2)
            W = n * H
        xi = int(np.clip(round((x0 + x1) / 2 - W / 2), 0, w - W))
        yi = int(np.clip(round((y0 + y1) / 2 - H / 2), 0, h - H))
        return -xi, -yi, W, H
    if n:
        W = n * H                    # ancho exacto: cada parte es un cuadrado H×H
    else:
        W, H = W + W % 2, H + H % 2
    ox = int(round((W - bw) / 2)) - x0
    oy = int(round((H - bh) / 2)) - y0
    return ox, oy, W, H


def mascara_suelo(imagen: np.ndarray, alfa: np.ndarray, op: OpcionesEstudio) -> np.ndarray | None:
    """Máscara del suelo real de la foto (modo "original"), con el ajuste manual
    de altura aplicado. None si no se distingue el suelo de la pared."""
    m = detectar_suelo(imagen, alfa)
    if m is None or op.suelo_ajuste == 0:
        return m
    _, y0, _, y1 = caja_sujeto(alfa, umbral=0.5)
    dy = int(round(op.suelo_ajuste * (y1 - y0)))  # + = la línea sube
    if dy > 0:
        m = np.vstack([m[dy:], np.repeat(m[-1:], dy, axis=0)])
    elif dy < 0:
        m = np.vstack([np.zeros((-dy, m.shape[1]), m.dtype), m[:dy]])
    return m


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

def componer(imagen: np.ndarray, alfa: np.ndarray, op: OpcionesEstudio,
             primer_plano: np.ndarray | None = None,
             suelo_mascara: np.ndarray | None = None) -> np.ndarray:
    """Compone el sujeto sobre el fondo de estudio. Devuelve uint8 RGB.

    ``primer_plano`` permite reutilizar el resultado de :func:`descontaminar` (la
    parte lenta) cuando sólo cambian los ajustes del fondo. ``suelo_mascara`` permite
    pasar la máscara del suelo real ya calculada (modo "original").
    """
    alfa = alfa.astype(np.float32)
    if op.suelo == "original":
        if suelo_mascara is None:
            suelo_mascara = mascara_suelo(imagen, alfa, op)
        if suelo_mascara is None:
            op = replace(op, suelo="estudio")  # no se distingue el suelo: estudio
    else:
        suelo_mascara = None
    if op.suelo != "estudio":
        op = replace(op, reflejo=0.0)  # el reflejo sólo tiene sentido en el suelo negro brillante
    if not op.descontaminar_bordes:
        frg = imagen.astype(np.float32) / 255.0
    elif primer_plano is not None:
        frg = primer_plano
    else:
        frg = descontaminar(imagen, alfa)
    frg = corregir_luz(frg, imagen, alfa, op.corregir_dominante)

    ox, oy, W, H = calcular_lienzo(alfa, op, dentro_de_la_foto=suelo_mascara is not None)
    a_l = np.zeros((H, W), np.float32)
    f_l = np.zeros((H, W, 3), np.float32)
    _pegar(a_l, alfa, ox, oy)
    _pegar(f_l, frg, ox, oy)

    geo = geometria_peana(a_l)
    geo["peanas"] = peanas(a_l)
    geo["alfa"] = a_l
    s_l = np.zeros((H, W), np.float32)
    if suelo_mascara is not None:
        # Suelo real: se conserva tal cual; sólo la pared pasa a estudio (sin
        # horizonte ni sombra generados: la foto ya tiene los suyos).
        fondo = fondo_estudio(H, W, geo, replace(op, horizonte=0.0, sombra=0.0, suelo="estudio"))
        _pegar(s_l, suelo_mascara, ox, oy)
        orig_f = np.zeros((H, W, 3), np.float32)
        _pegar(orig_f, imagen.astype(np.float32) / 255.0, ox, oy)
        s3 = s_l[..., None]
        fondo = fondo * (1 - s3) + _srgb_a_lineal(orig_f) * s3
    else:
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
        envolver = np.clip((1 - interior) * a_l * 2.0, 0, 1) * op.oscurecer_borde * (1 - s_l)
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
    if suelo_mascara is not None:
        # El suelo real conservado también es bit a bit el original.
        orig = np.zeros((H, W, 3), np.uint8)
        _pegar(orig, imagen, ox, oy)
        intacto = s_l >= 1.0
        out[intacto] = np.where((a_l[intacto] >= 1.0)[:, None], out[intacto], orig[intacto])
    return out


def recorte_transparente(imagen: np.ndarray, alfa: np.ndarray, descontaminar_bordes: bool = True) -> np.ndarray:
    """PNG RGBA del sujeto recortado (útil para otros montajes)."""
    frg = descontaminar(imagen, alfa) if descontaminar_bordes else imagen / 255.0
    rgba = np.dstack([np.round(np.clip(frg, 0, 1) * 255), np.round(alfa * 255)]).astype(np.uint8)
    return rgba
