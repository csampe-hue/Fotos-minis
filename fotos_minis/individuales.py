"""Fotos individuales de cada figura, recortadas de la foto de grupo ya procesada.

No se reprocesa nada: se localizan las peanas en la silueta del resultado y, para
cada una, se recorta de la imagen final un encuadre con la relación de aspecto
objetivo centrado en esa figura.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .composicion import (PROPORCIONES, OpcionesEstudio, _pegar, calcular_lienzo, mascara_peanas,
                          partes_carrusel, peanas)


@dataclass
class Figura:
    cx: float        # centro de la peana
    cy: float
    a: float         # semiejes de la peana
    b: float
    caja: tuple[int, int, int, int]   # x0, y0, x1, y1 de la figura completa


def _elipse(a: float, b: float) -> np.ndarray:
    a, b = max(int(round(a)), 2), max(int(round(b)), 1)
    k = np.zeros((2 * b + 1, 2 * a + 1), np.float32)
    cv2.ellipse(k, (a, b), (a, b), 0, 0, 360, 1.0, -1)
    return k


def estimar_peana(alfa: np.ndarray) -> tuple[float, float] | None:
    """Ancho y semieje vertical de una peana, aunque todas estén pegadas en fila.

    El borde inferior de la silueta está formado por arcos de elipse, uno por peana
    visible; donde dos peanas se tocan el borde forma una muesca hacia arriba. Se
    corta el contorno por los huecos y por esas muescas, y los arcos completos dan
    el ancho (entre sus extremos) y la profundidad del arco (el semieje vertical).
    """
    from scipy.signal import find_peaks

    sujeto = alfa > 0.5
    h, w = sujeto.shape
    hay = sujeto.any(axis=0)
    if hay.sum() < 8:
        return None
    fondo = np.where(hay, h - 1 - np.argmax(sujeto[::-1], axis=0), -1).astype(np.float32)
    arcos = []
    xs = np.flatnonzero(hay)
    tramos = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)     # separados por huecos
    for t in tramos:
        if len(t) < 6:
            continue
        perfil = cv2.GaussianBlur(fondo[t].reshape(1, -1), (0, 0), max(len(t) / 200, 1.0)).ravel()
        rango = perfil.max() - perfil.min()
        muescas, _ = find_peaks(-perfil, prominence=max(0.25 * rango, 2.0), distance=max(len(t) // 20, 3))
        cortes = [0, *muescas.tolist(), len(t) - 1]
        for i0, i1 in zip(cortes[:-1], cortes[1:]):
            if i1 - i0 < 6:
                continue
            seg = perfil[i0:i1 + 1]
            profundidad = seg.max() - (seg[0] + seg[-1]) / 2
            arcos.append((i1 - i0, max(profundidad, 1.0)))
    if not arcos:
        return None
    anchos = np.array([a for a, _ in arcos], np.float32)
    # Los arcos de peanas tapadas en parte son más cortos: se toman los más largos.
    ancho = float(np.percentile(anchos, 75))
    completos = [p for a, p in arcos if a >= 0.85 * ancho]
    vertical = float(np.median(completos)) if completos else ancho * 0.15
    return ancho, float(np.clip(vertical, 0.08 * ancho / 2, 0.8 * ancho / 2))


def localizar_peanas(alfa: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Centros y semiejes (cx, cy, a, b) de cada peana, aunque se toquen o solapen.

    Las peanas son elipses de tamaño conocido. Se colocan de una en una donde cubren
    más superficie de peana todavía sin explicar (cobertura voraz) y se para cuando
    una nueva elipse apenas aporta nada.
    """
    lista = peanas(alfa)
    # Dos estimaciones del tamaño de una peana que fallan en casos opuestos (piezas
    # sueltas de la silueta / arcos del borde inferior). Cuando fallan es porque juntan
    # varias peanas en una, así que la menor es la buena.
    estimada = estimar_peana(alfa)
    if estimada is not None and estimada[0] < float(np.median([g["ancho"] for g in lista])):
        ancho_e, vertical_e = estimada
        lista = [{"ancho": ancho_e, "inclinacion": vertical_e / (ancho_e / 2)}]
    mascara, ancho, vertical = mascara_peanas(alfa, lista)
    if not mascara.any():
        return []
    # Escala de trabajo: peanas de unos 40 px de ancho.
    s = min(1.0, 40.0 / max(ancho, 1.0))
    h, w = mascara.shape
    m = cv2.resize(mascara.astype(np.float32), (max(int(w * s), 1), max(int(h * s), 1)),
                   interpolation=cv2.INTER_AREA)
    a, b = ancho / 2 * s, vertical * s
    nucleo = _elipse(a, b)
    area = nucleo.sum()
    restante = m.copy()
    elegidas = []
    for _ in range(40):
        gana = cv2.filter2D(restante, -1, nucleo, borderType=cv2.BORDER_CONSTANT)
        cubre = cv2.filter2D(m, -1, nucleo, borderType=cv2.BORDER_CONSTANT)
        # Sólo centros donde la elipse está mayoritariamente sobre peana (oclusión parcial admitida).
        gana[cubre < 0.55 * area] = 0
        y, x = np.unravel_index(int(np.argmax(gana)), gana.shape)
        if gana[y, x] < 0.35 * area:
            break
        elegidas.append((x / s, y / s, ancho / 2, vertical))
        borrar = np.zeros_like(restante)
        cv2.ellipse(borrar, (int(x), int(y)), (max(int(a * 1.05), 1), max(int(b * 1.05), 1)), 0, 0, 360, 1.0, -1)
        restante *= 1 - borrar
    return elegidas


def figuras(alfa: np.ndarray) -> list[Figura]:
    """Figuras (una por peana) de izquierda a derecha, con la caja de cada una."""
    sujeto = alfa > 0.5
    h, w = sujeto.shape
    res = []
    for cx, cy, a, b in localizar_peanas(alfa):
        x0, x1 = int(max(cx - a, 0)), int(min(cx + a, w - 1))
        # Cima de la figura: se sube desde la peana por la silueta, columna a columna,
        # en la franja central; se limita a una altura razonable para una miniatura
        # (si otra figura de detrás continúa la silueta, no se la «come»).
        limite = int(max(cy - 2.2 * 2 * a, 0))
        cimas = []
        for x in range(int(max(cx - 0.6 * a, 0)), int(min(cx + 0.6 * a, w - 1)) + 1):
            y = int(min(cy, h - 1))
            if not sujeto[y, x]:
                continue
            while y > limite and sujeto[y - 1, x]:
                y -= 1
            cimas.append(y)
        y0 = int(min(cimas)) if cimas else int(max(cy - 2 * a, 0))   # el punto más alto (la cabeza)
        # Armas o brazos que sobresalen de la peana: se amplía lo que esté unido a la
        # silueta dentro de la altura de la figura (hasta medio ancho de peana más).
        franja = sujeto[y0:int(min(cy + b, h - 1)) + 1]
        cols = np.flatnonzero(franja[:, int(max(cx - 1.5 * a, 0)):int(min(cx + 1.5 * a, w - 1)) + 1].any(axis=0))
        if len(cols):
            base = int(max(cx - 1.5 * a, 0))
            x0 = min(x0, base + int(cols[0])) if base + cols[0] >= cx - 1.5 * a else x0
            x1 = max(x1, base + int(cols[-1]))
            x0, x1 = int(max(x0, cx - 1.5 * a)), int(min(x1, cx + 1.5 * a))
        res.append(Figura(cx, cy, a, b, (x0, y0, x1 + 1, int(min(cy + b, h - 1)) + 1)))
    res.sort(key=lambda f: (f.cx, f.cy))
    return res


def encuadre(caja: tuple[int, int, int, int], prop: float, margen: float,
             W: int, H: int) -> tuple[int, int, int, int]:
    """Ventana (x0, y0, x1, y1) con proporción ``prop`` (ancho/alto) que contiene la
    caja con margen, centrada en ella y sin salirse de la imagen W×H."""
    x0, y0, x1, y1 = caja
    bw, bh = x1 - x0, y1 - y0
    m = margen * bh
    ww, hh = bw + 2 * m, bh + 2 * m
    if ww / hh < prop:
        ww = hh * prop
    else:
        hh = ww / prop
    escala = min(1.0, W / ww, H / hh)          # no puede ser mayor que la foto
    ww, hh = ww * escala, hh * escala
    iw, ih = int(round(ww)), int(round(hh))
    if prop == 1.0:
        iw = ih = min(iw, ih)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ix = int(np.clip(round(cx - iw / 2), 0, W - iw))
    iy = int(np.clip(round(cy - ih / 2), 0, H - ih))
    return ix, iy, ix + iw, iy + ih


def proporcion_objetivo(op: OpcionesEstudio, alfa: np.ndarray, W: int, H: int) -> float:
    """Relación de aspecto de cada foto individual: 1:1 en modo carrusel (como cada
    parte), la elegida en «Proporción», o la de la foto original."""
    if partes_carrusel(alfa, op):
        return 1.0
    return PROPORCIONES.get(op.proporcion) or (alfa.shape[1] / alfa.shape[0])


def recortes(salida: np.ndarray, alfa: np.ndarray, op: OpcionesEstudio) -> list[np.ndarray]:
    """Recortes individuales de la imagen final ``salida`` (sólo si hay 2 o más figuras).

    ``alfa`` es la máscara de la foto original: se recoloca en el lienzo del
    resultado con el mismo cálculo de encuadre que usó la composición.
    """
    H, W = salida.shape[:2]
    lienzo = None
    for dentro in (False, True):
        ox, oy, w_, h_ = calcular_lienzo(alfa, op, dentro_de_la_foto=dentro)
        if (w_, h_) == (W, H):
            lienzo = np.zeros((H, W), np.float32)
            _pegar(lienzo, alfa, ox, oy)
            break
    if lienzo is None:
        return []
    figs = figuras(lienzo)
    if len(figs) < 2:
        return []
    prop = proporcion_objetivo(op, alfa, W, H)
    res = []
    for f in figs:
        x0, y0, x1, y1 = encuadre(f.caja, prop, op.margen, W, H)
        res.append(salida[y0:y1, x0:x1].copy())
    return res


def nombre_individual(base: str, i: int, ext: str = ".png") -> str:
    return f"{base}_figura{i}{ext}"


def borrar_individuales(carpeta: Path, base: str) -> None:
    """Elimina las fotos individuales de pruebas anteriores de la misma foto."""
    for f in carpeta.glob(f"{base}_figura*"):
        resto = f.stem[len(base) + len("_figura"):]
        if resto.isdigit():
            f.unlink(missing_ok=True)
