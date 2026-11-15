# Fotos Minis · Fondo negro de estudio

Sustituye el fondo de la foto de una figura o miniatura por un **negro de estudio**,
conservando intactos el sujeto y su peana: rostro, cuerpo, ropa, peinado, pose,
proporciones y detalles finos.

## Qué hace

1. **Segmentación** del sujeto con IA (BiRefNet por defecto, ISNet como alternativa
   rápida), vía [rembg](https://github.com/danielgatis/rembg). Los modelos se
   descargan solos la primera vez.
2. **Refinado de la máscara**: elimina motas del fondo, puede rellenar huecos pequeños
   y ajusta el contorno a los bordes reales de la foto con un filtro guiado (lanzas,
   pelo, capas, filo de la peana).
3. **Limpieza del contorno**: recalcula el color de los píxeles semitransparentes del
   borde para quitar el color del fondo antiguo (sin halos claros sobre el negro).
4. **Entorno de estudio**:
   - negro de fondo con foco suave detrás de la figura y viñeteado;
   - sombra de contacto elíptica bajo la peana, en la perspectiva de la cámara;
   - reflejo opcional sobre suelo negro brillante (metacrilato), simétrico respecto a
     la línea de apoyo de la peana;
   - el negro "envuelve" ligeramente el contorno, como en una foto real;
   - corrección opcional de la dominante de color de la luz original (medida en el
     fondo antiguo).
5. **Encuadre**: centra la figura con el margen y la proporción que elijas
   **sin reescalarla**.

**Garantía de fidelidad:** los píxeles opacos del sujeto se copian bit a bit del
original. Sólo cambian los del borde semitransparente y, si lo activas, la corrección
de dominante de color (por defecto a 0).

## Instalación

Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Memoria: BiRefNet necesita unos **8 GB de RAM** en CPU (≈20 s por foto). Si tu equipo
tiene menos, elige ISNet (≈1 GB, ≈2 s por foto). Con GPU NVIDIA instala
`rembg[gpu]` en lugar de `rembg[cpu]`.

## Uso

### Interfaz web

```bash
python app.py
```

Abre <http://127.0.0.1:7860>, sube la foto y ajusta los controles; el resultado se
actualiza al momento (la segmentación se calcula una vez por foto). También permite
procesar varias fotos y descargarlas en un ZIP.

### Línea de comandos

```bash
# Una foto o una carpeta entera
python -m fotos_minis fotos/ -o salida/

# Con reflejo, formato cuadrado y PNG transparente adicional
python -m fotos_minis mini.jpg --reflejo 0.35 --proporcion 1:1 --recorte

# Modelo rápido
python -m fotos_minis fotos/ --modelo isnet-general-use
```

`python -m fotos_minis -h` muestra todas las opciones.

### Desde Python

```python
from fotos_minis import procesar, OpcionesEstudio

r = procesar("mini.jpg", OpcionesEstudio(reflejo=0.3, proporcion="4:5"))
r.estudio.save("mini_estudio.png")
r.recorte.save("mini_recorte.png")   # RGBA
```

## Consejos para mejores resultados

- Fotografía la figura completa dentro del encuadre, peana incluida.
- Un fondo liso y que contraste con la figura ayuda mucho al recorte.
- Si zonas muy oscuras de la figura salen agujereadas, sube
  «Rellenar huecos pequeños de la máscara».
- Revisa la máscara en la interfaz: blanco = se conserva, negro = se sustituye.

## Pruebas

```bash
pip install pytest
python -m pytest
```
