"""Super-resolución opcional con Real-ESRGAN (realesr-general-x4v3, ONNX).

Aumenta la resolución de las fotos individuales, que al ser recortes de la foto de
grupo pueden quedar pequeñas y borrosas. Está **desactivada por defecto**
(``OpcionesEstudio.superres = 0``): sólo se aplica si se pide ×2 o ×4.

El modelo amplía ×4. Para ×2 cada tesela ×4 se reduce a la mitad con un filtro de área
(más nítido que ampliar ×2 directamente) antes de colocarla en el resultado, así que
nunca se guarda la imagen ×4 entera en memoria. Si el resultado pasaría de
``LADO_MAXIMO`` se amplía al factor entero más pequeño que llega al tope y después se
reduce, con lo que la imagen intermedia nunca supera ~2·LADO_MAXIMO de lado.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODELO = Path(__file__).resolve().parent.parent / "recursos" / "modelos" / "realesr_general_x4v3.onnx"
LADO_MAXIMO = 4096          # tope del lado mayor del resultado (memoria y tamaño de fichero)
TESELA = 256                # se procesa por teselas para acotar la memoria (múltiplo de 2)
SOLAPE = 16
FACTOR_MINIMO = 1.05        # por debajo no merece la pena: se devuelve la imagen tal cual

log = logging.getLogger(__name__)


class ErrorSuperres(RuntimeError):
    """La super-resolución no se ha podido aplicar (el mensaje es apto para el usuario)."""


@lru_cache(maxsize=1)
def _sesion():
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover - onnxruntime llega con rembg
        raise ErrorSuperres("falta la librería onnxruntime (pip install -r requirements.txt)") from e
    opciones = ort.SessionOptions()
    opciones.enable_cpu_mem_arena = False
    try:
        return ort.InferenceSession(str(MODELO), opciones, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise ErrorSuperres(f"no se pudo cargar el modelo {MODELO.name}: {e}") from e


def disponible() -> bool:
    return MODELO.is_file()


def factor_efectivo(forma: tuple[int, ...], factor: int) -> float:
    """Factor realmente aplicado: el pedido, limitado para no pasar de LADO_MAXIMO."""
    return float(min(factor, LADO_MAXIMO / max(forma[:2])))


def _rgb(img: np.ndarray) -> np.ndarray:
    """La imagen como RGB uint8 (acepta gris, RGB o RGBA)."""
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _ampliar_ia(img: np.ndarray, factor: int) -> np.ndarray:
    """Ampliación ×factor (2 o 4) por teselas con solape (sin costuras).

    Cada tesela se amplía ×4 con el modelo y, para ×2, se reduce a la mitad con
    INTER_AREA. Como las teselas empiezan en coordenadas pares y la reducción es
    exactamente 2:1, el resultado es idéntico al de reducir la imagen ×4 completa, pero
    sólo hace falta memoria para la salida final (uint8) y una tesela.
    """
    if factor not in (2, 4):
        raise ValueError(f"factor de super-resolución no válido: {factor}")
    h, w = img.shape[:2]
    out = np.empty((h * factor, w * factor, 3), np.uint8)
    sesion = _sesion()
    nombre = sesion.get_inputs()[0].name
    for y0 in range(0, h, TESELA):
        for x0 in range(0, w, TESELA):
            y1, x1 = min(y0 + TESELA, h), min(x0 + TESELA, w)
            ya, xa = max(y0 - SOLAPE, 0), max(x0 - SOLAPE, 0)
            yb, xb = min(y1 + SOLAPE, h), min(x1 + SOLAPE, w)
            tesela = (img[ya:yb, xa:xb].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            y = sesion.run(None, {nombre: np.ascontiguousarray(tesela)})[0][0].transpose(1, 2, 0)
            y = y[(y0 - ya) * 4:(y1 - ya) * 4, (x0 - xa) * 4:(x1 - xa) * 4]
            y = np.clip(np.round(y * 255), 0, 255).astype(np.uint8)
            if factor == 2:
                y = cv2.resize(y, ((x1 - x0) * 2, (y1 - y0) * 2), interpolation=cv2.INTER_AREA)
            out[y0 * factor:y1 * factor, x0 * factor:x1 * factor] = y
    return out


def ampliar(img: np.ndarray, factor: int) -> np.ndarray:
    """Devuelve la imagen ampliada ×factor (2 o 4) con IA, como RGB uint8.

    Sin cambios si ``factor <= 1`` o si la imagen ya es tan grande que el tope de
    ``LADO_MAXIMO`` deja el factor por debajo de ``FACTOR_MINIMO``.

    Lanza :class:`ErrorSuperres` si falta el modelo, no se puede cargar, o la
    inferencia falla (p. ej. por falta de memoria); quien llama decide si seguir con la
    imagen original.
    """
    if factor <= 1:
        return img
    if factor not in (2, 4):
        raise ValueError(f"factor de super-resolución no válido: {factor} (usa 2 o 4)")
    f = factor_efectivo(img.shape, factor)
    if f < FACTOR_MINIMO:
        return img
    if not disponible():
        raise ErrorSuperres(f"no se encuentra el modelo {MODELO} (vuelve a descargar el programa)")
    rgb = _rgb(img)
    base = 2 if f <= 2 else 4                          # factor entero más pequeño que llega a f
    try:
        grande = _ampliar_ia(rgb, base)
    except ErrorSuperres:
        raise
    except MemoryError as e:
        raise ErrorSuperres("no hay memoria suficiente; prueba con ×2") from e
    except Exception as e:
        raise ErrorSuperres(f"falló la inferencia del modelo: {e}") from e
    if abs(f - base) < 1e-6:
        return grande
    h, w = rgb.shape[:2]
    return cv2.resize(grande, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)


def ampliar_o_aviso(img: np.ndarray, factor: int) -> tuple[np.ndarray, str | None]:
    """Como :func:`ampliar`, pero sin lanzar: si falla devuelve la imagen original y un
    aviso para mostrar al usuario (y lo deja en el registro)."""
    try:
        return ampliar(img, factor), None
    except ErrorSuperres as e:
        log.warning("Super-resolución desactivada para esta imagen: %s", e, exc_info=e.__cause__ is not None)
        return img, f"No se pudo aplicar la super-resolución ({e}); se guarda sin ampliar."
