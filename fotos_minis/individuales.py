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

from .composicion import (PROPORCIONES, OpcionesEstudio, _pegar, caja_sujeto, calcular_lienzo,
                          geometria_peana, mascara_peanas, partes_carrusel, piezas_apoyadas)


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
        sigma = max(len(t) / 200, 1.0)
        perfil = cv2.GaussianBlur(fondo[t].reshape(1, -1), (0, 0), sigma).ravel()
        rango = perfil.max() - perfil.min()
        muescas, _ = find_peaks(-perfil, prominence=max(0.25 * rango, 2.0), distance=max(len(t) // 20, 3))
        # Escalones: donde una peana delantera termina y asoma una de detrás, el borde
        # salta de golpe hacia arriba (no forma muesca). También se corta allí.
        k = max(3, int(round(2 * sigma)))                      # el suavizado reparte el salto
        salto = np.abs(perfil[2 * k:] - perfil[:-2 * k])
        escalones, _ = find_peaks(salto, height=max(0.2 * rango, 6.0), distance=k + 1)
        cortes = sorted({0, *muescas.tolist(), *(escalones + k).tolist(), len(t) - 1})
        for i0, i1 in zip(cortes[:-1], cortes[1:]):
            if i1 - i0 < 6:
                continue
            seg = perfil[i0:i1 + 1]
            profundidad = seg.max() - (seg[0] + seg[-1]) / 2
            arcos.append((i1 - i0, max(profundidad, 1.0)))
    if not arcos:
        return None
    # Los trozos muy cortos (esquinas, un pico de peana asomando) no informan del ancho.
    largo = max(a for a, _ in arcos)
    arcos = [(a, p) for a, p in arcos if a >= 0.3 * largo]
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

    Se trabaja por piezas de la silueta (sin lanzas ni otras estructuras finas, que
    pueden unir figuras separadas): cada pieza grande mide su propio tamaño de peana,
    así que en una foto con peanas de 25 y de 50 mm separadas cada una se mide bien.
    Las piezas pequeñas usan el tamaño típico del resto.
    """
    piezas = piezas_apoyadas(alfa)
    if not piezas:
        return []
    areas = [p.sum() for p in piezas]
    tamanos = []
    for p in piezas:
        a_p = np.where(p, alfa, 0).astype(np.float32)
        g = geometria_peana(a_p)
        ancho, incl = g["ancho"], g["inclinacion"]
        # Dos estimaciones del tamaño de una peana que fallan en casos opuestos (pieza
        # entera / arcos del borde inferior). Cuando fallan es porque juntan varias
        # peanas en una, así que la menor es la buena.
        estimada = estimar_peana(a_p)
        if estimada is not None and estimada[0] < ancho:
            ancho, incl = estimada[0], estimada[1] / (estimada[0] / 2)
        tamanos.append((ancho, incl))
    tipico = (float(np.median([t[0] for t in tamanos])), float(np.median([t[1] for t in tamanos])))
    elegidas = []
    for p, area, t in zip(piezas, areas, tamanos):
        if area < 0.15 * max(areas):
            t = tipico
        elegidas += _cubrir(np.where(p, alfa, 0).astype(np.float32), *t)
    return _descartar_falsas(elegidas, alfa)


def _cubrir(alfa: np.ndarray, ancho: float, inclinacion: float) -> list[tuple[float, float, float, float]]:
    """Cobertura voraz de las peanas de una pieza con elipses del tamaño dado."""
    ys, xs = np.nonzero(alfa > 0.5)
    if len(ys) == 0:
        return []
    h_, w_ = alfa.shape
    pad = int(ancho) + 4
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h_)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w_)
    mascara, ancho, vertical = mascara_peanas(alfa[y0:y1, x0:x1], [{"ancho": ancho, "inclinacion": inclinacion}])
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
    # Dos peanas redondas no pueden ocupar el mismo sitio en la mesa: en la imagen sus
    # centros distan al menos dos semiejes (en la escala de cada eje). Se prohíben los
    # centros a menos de 1,6 semiejes de una peana ya colocada (margen para el error
    # de medida). Evita peanas fantasma en una capa o un manto que llega a la peana.
    prohibido = np.zeros_like(m)
    elegidas = []
    for _ in range(40):
        gana = cv2.filter2D(restante, -1, nucleo, borderType=cv2.BORDER_CONSTANT)
        cubre = cv2.filter2D(m, -1, nucleo, borderType=cv2.BORDER_CONSTANT)
        # Sólo centros donde la elipse está mayoritariamente sobre peana (oclusión parcial admitida).
        gana[cubre < 0.55 * area] = 0
        gana[prohibido > 0] = 0
        y, x = np.unravel_index(int(np.argmax(gana)), gana.shape)
        if gana[y, x] < 0.35 * area:
            break
        elegidas.append((x / s + x0, y / s + y0, ancho / 2, vertical))
        borrar = np.zeros_like(restante)
        cv2.ellipse(borrar, (int(x), int(y)), (max(int(a * 1.05), 1), max(int(b * 1.05), 1)), 0, 0, 360, 1.0, -1)
        restante *= 1 - borrar
        cv2.ellipse(prohibido, (int(x), int(y)), (max(int(a * 1.6), 1), max(int(b * 1.6), 1)), 0, 0, 360, 1.0, -1)
    return elegidas


def _descartar_falsas(bases, alfa):
    """Quita «peanas» que en realidad son el cuerpo de una figura (p. ej. torso y una
    pierna levantada en horizontal forman una forma ancha y plana). Una peana real
    tapada por otra queda como mucho a ~2 alturas de peana por encima de ella; si hay
    otra peana casi en la misma vertical, unida a ella y mucho más abajo, es falsa."""
    etiquetas = cv2.connectedComponents((alfa > 0.5).astype(np.uint8), connectivity=8)[1]
    h, w = alfa.shape

    def pieza(x, y):
        return etiquetas[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))]

    buenas = []
    for x, y, a, b in bases:
        falsa = any(y2 - y > 2.8 * b and abs(x2 - x) < 0.5 * a and pieza(x, y) and pieza(x, y) == pieza(x2, y2)
                    for x2, y2, _, _ in bases)
        if not falsa:
            buenas.append((x, y, a, b))
    return buenas


def mascaras_figuras(alfa: np.ndarray, imagen: np.ndarray | None = None) -> list[np.ndarray]:
    """Máscara alfa de cada figura (de izquierda a derecha), o [] si hay menos de dos.

    Cada figura es todo lo que está unido a su peana en la silueta (brazos levantados,
    armas...). Si varias figuras forman una sola pieza porque se solapan, se separan
    con una divisoria (watershed) sobre la foto, sembrada en cada peana.
    """
    bases = localizar_peanas(alfa)
    if len(bases) < 2:
        return []
    sujeto = (alfa > 0.5).astype(np.uint8)
    h, w = sujeto.shape
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(sujeto, connectivity=8)
    propias = np.zeros((h, w), np.int32)          # 1..len(bases): figura de cada píxel

    def comp_de(cx, cy):
        x, y = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
        if etiquetas[y, x]:
            return etiquetas[y, x]
        ventana = etiquetas[max(y - 5, 0):y + 6, max(x - 5, 0):x + 6]
        valores = ventana[ventana > 0]
        return int(np.bincount(valores).argmax()) if len(valores) else 0

    por_comp: dict[int, list[int]] = {}
    for i, (cx, cy, a, b) in enumerate(bases):
        por_comp.setdefault(comp_de(cx, cy), []).append(i)
    for k, indices in por_comp.items():
        if k == 0:
            continue
        zona = etiquetas == k
        if len(indices) == 1:
            propias[zona] = indices[0] + 1
            continue
        marcas = np.zeros((h, w), np.int32)
        marcas[~zona] = len(bases) + 1                         # fuera de la pieza
        # Semillas: la peana y la «columna» del cuerpo encima de ella. Se dibujan de
        # atrás adelante: donde se solapan manda la figura delantera (la que tapa).
        for i in sorted(indices, key=lambda j: bases[j][1]):
            cx, cy, a, b = bases[i]
            semilla = np.zeros((h, w), np.uint8)
            cv2.ellipse(semilla, (int(cx), int(cy)), (max(int(a * 0.6), 1), max(int(b * 0.6), 1)),
                        0, 0, 360, 1, -1)
            cv2.rectangle(semilla, (int(cx - 0.15 * a), int(cy - 1.5 * a)), (int(cx + 0.15 * a), int(cy)), 1, -1)
            # La figura de detrás no siembra donde está el cuerpo de una de delante.
            for j in indices:
                fx, fy, fa, fb = bases[j]
                if fy > cy + 0.3 * b:
                    cv2.rectangle(semilla, (int(fx - 0.6 * fa), int(fy - 2.4 * fa)),
                                  (int(fx + 0.6 * fa), int(fy + fb)), 0, -1)
            marcas[(semilla > 0) & zona] = i + 1
        guia = imagen if imagen is not None else np.dstack([(alfa * 255).astype(np.uint8)] * 3)
        cv2.watershed(np.ascontiguousarray(guia[..., :3]).astype(np.uint8), marcas)
        lineas = (marcas == -1) & zona
        marcas[lineas] = 0
        marcas = np.where(marcas > len(bases), 0, marcas)
        # Las líneas divisorias pasan a la figura vecina más cercana.
        if lineas.any():
            dil = cv2.dilate(marcas.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
            marcas[lineas] = dil[lineas]
        # Trozos que la divisoria deja sueltos (p. ej. un casco del vecino): cada figura
        # conserva sólo la parte unida a su peana; el resto pasa a la figura que toca.
        for i in indices:
            cx, cy, a, b = bases[i]
            n_t, et_t = cv2.connectedComponents((marcas == i + 1).astype(np.uint8), connectivity=8)
            if n_t > 2:
                x, y = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
                propia = et_t[y, x] or np.bincount(et_t[et_t > 0]).argmax()
                marcas[(et_t > 0) & (et_t != propia)] = 0
        for _ in range(200):
            huecos = zona & (marcas <= 0)
            if not huecos.any():
                break
            dil = cv2.dilate(np.where(marcas > 0, marcas, 0).astype(np.float32),
                             np.ones((3, 3), np.uint8)).astype(np.int32)
            marcas[huecos & (dil > 0)] = dil[huecos & (dil > 0)]
        propias[zona] = np.where(marcas[zona] > 0, marcas[zona], 0)
    # Piezas sueltas sin peana (p. ej. la punta de un arma separada): a la figura más cercana.
    for k in range(1, n):
        if k in por_comp:
            continue
        ys, xs = np.nonzero(etiquetas == k)
        cx, cy = xs.mean(), ys.mean()
        j = int(np.argmin([abs(cx - b[0]) + 0.3 * abs(cy - b[1]) for b in bases]))
        propias[etiquetas == k] = j + 1

    orden = sorted(range(len(bases)), key=lambda i: (bases[i][0], bases[i][1]))
    res = []
    for i in orden:
        m = cv2.dilate((propias == i + 1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        # Las otras figuras no entran aunque la dilatación las roce.
        otras = (propias > 0) & (propias != i + 1)
        res.append(np.where(m & ~otras, alfa, 0).astype(np.float32))
    return [m for m in res if (m > 0.5).any()]


def figuras(alfa: np.ndarray) -> list[Figura]:
    """Figuras de izquierda a derecha con la caja completa de cada una."""
    res = []
    bases = sorted(localizar_peanas(alfa), key=lambda b: (b[0], b[1]))
    for (cx, cy, a, b), m in zip(bases, mascaras_figuras(alfa)):
        x0, y0, x1, y1 = caja_sujeto(m, umbral=0.5)
        res.append(Figura(cx, cy, a, b, (x0, y0, x1, y1)))
    return res


def encuadre_centrado(caja: tuple[int, int, int, int], prop: float, margen: float) -> tuple[int, int, int, int]:
    """Ventana con proporción ``prop`` centrada exactamente en la caja, con margen
    (medido como en el encuadre normal). Puede salirse de la imagen."""
    x0, y0, x1, y1 = caja
    bw, bh = x1 - x0, y1 - y0
    m = margen * max(bw, bh)
    ww, hh = bw + 2 * m, bh + 2 * m
    if ww / hh < prop:
        ww = hh * prop
    else:
        hh = ww / prop
    iw, ih = int(round(ww)), int(round(hh))
    if abs(prop - 1.0) < 1e-6:
        iw = ih = max(iw, ih)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ix, iy = int(round(cx - iw / 2)), int(round(cy - ih / 2))
    return ix, iy, ix + iw, iy + ih


def proporcion_objetivo(op: OpcionesEstudio, alfa: np.ndarray) -> str:
    """Proporción de cada foto individual (clave de PROPORCIONES u «original»):
    1:1 en modo carrusel (como cada parte); si no, la elegida en «Proporción»."""
    if partes_carrusel(alfa, op):
        return "1:1"
    return op.proporcion


def individuales(salida: np.ndarray, alfa: np.ndarray, op: OpcionesEstudio, imagen: np.ndarray,
                 primer_plano: np.ndarray | None = None,
                 avisos: list[str] | None = None) -> tuple[list[tuple[np.ndarray, str]], int]:
    """Una imagen por figura, centrada y entera: ([(imagen, "recorte" | "nuevo fondo")], tapadas).

    Si ``op.superres`` es 2 o 4 se amplían con IA; si eso falla, la foto se deja sin
    ampliar y el motivo se añade a ``avisos`` (una vez, aunque falle en varias figuras).

    Las figuras casi ocultas tras otras (se ve menos de un tercio de lo que se ve de
    una figura típica) no se pueden mostrar enteras: se omiten y se cuentan en ``tapadas``.

    Si el encuadre centrado de la figura cabe dentro de la foto de grupo y no entra
    ninguna otra figura, se recorta de ``salida`` (sin reprocesar). Si no, esa figura
    se vuelve a componer sola sobre un fondo nuevo (reutilizando su máscara: no se
    segmenta otra vez).
    """
    from dataclasses import replace

    from .composicion import componer

    mascaras = mascaras_figuras(alfa, imagen)
    if len(mascaras) < 2:
        return [], 0
    areas = np.array([(m > 0.5).sum() for m in mascaras], np.float32)
    visibles = [m for m, ar in zip(mascaras, areas) if ar >= 0.33 * np.median(areas)]
    tapadas = len(mascaras) - len(visibles)
    mascaras = visibles
    H, W = salida.shape[:2]
    desplaz = None
    for dentro in (False, True):
        ox, oy, w_, h_ = calcular_lienzo(alfa, op, dentro_de_la_foto=dentro)
        if (w_, h_) == (W, H):
            desplaz = (ox, oy)
            break
    clave = proporcion_objetivo(op, alfa)
    prop = PROPORCIONES.get(clave) or (alfa.shape[1] / alfa.shape[0])
    if op.nitidez > 0:
        # Se restaura una vez con la máscara de todas las figuras (como en la foto de
        # grupo) y no otra vez por figura.
        from .nitidez import aplicar
        imagen, primer_plano = aplicar(imagen, alfa, op.nitidez, primer_plano)
    op_ind = replace(op, carrusel=0, individuales=False, proporcion=clave, reencuadrar=True, nitidez=0.0)
    total = alfa > 0.5
    res = []
    for m in mascaras:
        propia = m > 0.5
        if desplaz is not None:
            x0, y0, x1, y1 = caja_sujeto(m, umbral=0.5)
            ox, oy = desplaz
            vx0, vy0, vx1, vy1 = encuadre_centrado((x0 + ox, y0 + oy, x1 + ox, y1 + oy), prop, op.margen)
            dentro = vx0 >= 0 and vy0 >= 0 and vx1 <= W and vy1 <= H
            if dentro:
                otras = np.zeros((H, W), bool)
                _pegar(otras, total & ~propia, ox, oy)
                if otras[vy0:vy1, vx0:vx1].sum() <= 0.0005 * (vx1 - vx0) * (vy1 - vy0):
                    res.append((_mejorar(salida[vy0:vy1, vx0:vx1].copy(), op, avisos), "recorte"))
                    continue
        # Fondo nuevo: la figura sola. Con el suelo real de la foto, las otras figuras
        # se borran de él (inpainting) para que no reaparezcan.
        img_i = imagen
        if op.suelo == "original":
            otras_orig = cv2.dilate((total & ~propia).astype(np.uint8), np.ones((9, 9), np.uint8))
            img_i = cv2.inpaint(np.ascontiguousarray(imagen[..., :3]), otras_orig, 7, cv2.INPAINT_TELEA)
        res.append((_mejorar(componer(img_i, m, op_ind, primer_plano), op, avisos), "nuevo fondo"))
    return res, tapadas


def _mejorar(img: np.ndarray, op: OpcionesEstudio, avisos: list[str] | None = None) -> np.ndarray:
    """Super-resolución opcional (Real-ESRGAN) de una foto individual. Desactivada por
    defecto; un fallo nunca interrumpe el proceso: se sigue con la foto sin ampliar."""
    if op.superres not in (2, 4):
        return img
    from .superres import ampliar_o_aviso
    img, aviso = ampliar_o_aviso(img, op.superres)
    if aviso and avisos is not None and aviso not in avisos:
        avisos.append(aviso)
    return img


def recortes(salida: np.ndarray, alfa: np.ndarray, op: OpcionesEstudio,
             imagen: np.ndarray | None = None, primer_plano: np.ndarray | None = None) -> list[np.ndarray]:
    """Sólo las imágenes de :func:`individuales` (compatibilidad)."""
    if imagen is None:
        imagen = salida if salida.shape[:2] == alfa.shape else np.zeros((*alfa.shape, 3), np.uint8)
    return [im for im, _ in individuales(salida, alfa, op, imagen, primer_plano)[0]]


def nombre_individual(base: str, i: int, ext: str = ".png") -> str:
    return f"{base}_figura{i}{ext}"


def borrar_individuales(carpeta: Path, base: str) -> None:
    """Elimina las fotos individuales de pruebas anteriores de la misma foto."""
    for f in carpeta.glob(f"{base}_figura*"):
        resto = f.stem[len(base) + len("_figura"):]
        if resto.isdigit():
            f.unlink(missing_ok=True)
