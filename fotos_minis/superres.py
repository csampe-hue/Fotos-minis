"""Super-resolución opcional con Real-ESRGAN (realesr-general-x4v3, ONNX).

Aumenta la resolución de las fotos individuales, que al ser recortes de la foto de
grupo pueden quedar pequeñas y borrosas. El modelo amplía ×4; para ×2 se reduce después
con un filtro de área (más nítido que ampliar ×2 directamente).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODELO = Path(__file__).resolve().parent.parent / "recursos" / "modelos" / "realesr_general_x4v3.onnx"
LADO_MAXIMO = 4096          # tope del lado mayor del resultado (memoria y tamaño de fichero)
TESELA = 256                # se procesa por teselas para acotar la memoria
SOLAPE = 16


@lru_cache(maxsize=1)
def _sesion():
    import onnxruntime as ort

    opciones = ort.SessionOptions()
    opciones.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(MODELO), opciones, providers=["CPUExecutionProvider"])


def disponible() -> bool:
    return MODELO.is_file()


def _x4(img: np.ndarray) -> np.ndarray:
    """Ampliación ×4 por teselas con solape (sin costuras)."""
    h, w = img.shape[:2]
    x = img.astype(np.float32) / 255.0
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    sesion = _sesion()
    for y0 in range(0, h, TESELA):
        for x0 in range(0, w, TESELA):
            y1, x1 = min(y0 + TESELA, h), min(x0 + TESELA, w)
            ya, xa = max(y0 - SOLAPE, 0), max(x0 - SOLAPE, 0)
            yb, xb = min(y1 + SOLAPE, h), min(x1 + SOLAPE, w)
            tesela = x[ya:yb, xa:xb].transpose(2, 0, 1)[None]
            y = sesion.run(None, {"x": np.ascontiguousarray(tesela)})[0][0].transpose(1, 2, 0)
            out[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = y[(y0 - ya) * 4:(y1 - ya) * 4, (x0 - xa) * 4:(x1 - xa) * 4]
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8)


def factor_efectivo(forma: tuple[int, int], factor: int) -> float:
    """Factor realmente aplicado: el pedido, limitado para no pasar de LADO_MAXIMO."""
    return float(min(factor, LADO_MAXIMO / max(forma[:2])))


def ampliar(img: np.ndarray, factor: int) -> np.ndarray:
    """Devuelve la imagen ampliada ×factor (2 o 4) con IA. Sin cambios si factor<=1,
    si no hay modelo o si la imagen ya es grande."""
    if factor <= 1 or not disponible():
        return img
    f = factor_efectivo(img.shape, factor)
    if f <= 1.05:
        return img
    grande = _x4(img[..., :3])
    if abs(f - 4) < 1e-6:
        return grande
    h, w = img.shape[:2]
    return cv2.resize(grande, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
