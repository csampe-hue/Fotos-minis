# Fotos Minis · Fondo negro de estudio

Sustituye el fondo de la foto de una figura o miniatura por un **negro de estudio**,
conservando intactos el sujeto y su peana: rostro, cuerpo, ropa, peinado, pose,
proporciones y detalles finos.

## Qué hace

1. **Segmentación** del sujeto con IA vía [rembg](https://github.com/danielgatis/rembg).
   Por defecto se combinan dos modelos (BiRefNet + ISNet): en pruebas con fotos reales,
   BiRefNet llegó a descartar una peana de arena gris sobre un tapete gris, e ISNet
   la recuperaba; la unión de ambos acierta en todos los casos probados. Los modelos
   se descargan solos la primera vez.
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
     fondo antiguo);
   - **horizonte** opcional: paso visible del suelo a la pared del fondo. Su altura se
     calcula con la perspectiva de la peana (una peana redonda vista desde arriba es una
     elipse, cuya proporción da la inclinación de la cámara) y la línea se desenfoca más
     cuanto más lejos está la pared.
5. **Encuadre**: centra la figura en horizontal y en vertical, con el margen y la proporción que elijas
   **sin reescalarla**.

**Garantía de fidelidad:** los píxeles opacos del sujeto se copian bit a bit del
original. Sólo cambian los del borde semitransparente y, si lo activas, la corrección
de dominante de color (por defecto a 0).

## Instalación

Python 3.11 o superior (recomendado 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Memoria: el modo combinado (por defecto) necesita unos **9 GB de RAM** en CPU
(≈25 s por foto). Si tu equipo tiene menos, elige ISNet (≈1 GB, ≈2 s por foto). Con GPU NVIDIA instala
`rembg[gpu]` en lugar de `rembg[cpu]`.

## Uso

### Arranque con doble clic (Mac)

Haz doble clic en **`Iniciar Fotos Minis.command`**. La primera vez prepara el entorno
(instala uv, Python 3.12 y las librerías; tarda unos minutos) y después arranca la
aplicación y abre el navegador. Para cerrarla, pulsa **Cerrar aplicación** al final de la
página (o cierra la ventana de Terminal).

Si macOS lo bloquea por ser de un «desarrollador no identificado», haz clic derecho
sobre el fichero → **Abrir** → **Abrir** (sólo hace falta la primera vez).

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
python -m fotos_minis mis_fotos/          # resultados en la carpeta de guardado

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

## Dónde se guardan los resultados

Todo se guarda automáticamente en la **carpeta de guardado**, que se elige bajo el visor:
escribe o pega una ruta, o pulsa **Elegir…** para abrir el selector de carpetas del
sistema; **Abrir** la muestra en el Finder. Se recuerda la última utilizada
(`~/.fotos-minis/preferencias.json`) y es la que aparece al abrir la aplicación. Si no
has elegido ninguna, se usa `fotos` dentro de la carpeta desde la que se ejecuta la
aplicación. Si la carpeta recordada deja de estar disponible (p. ej. un disco externo
desconectado), se usa esa carpeta `fotos` por defecto.

- **Una foto**: `<carpeta>/<nombre-original>_estudio.png`. Si sigues probando ajustes con
  la misma foto original, el fichero se sobrescribe y sólo queda el último resultado; otra
  foto original crea su propio fichero.
- **Varias fotos**: el ZIP del lote, `<carpeta>/fotos_estudio_<fecha_hora>.zip`.
- **Línea de comandos**: usa la misma carpeta por defecto (cámbiala con `-o`).

La carpeta `fotos` del proyecto está excluida de git, así que tus imágenes no se suben al
repositorio.

## Tipos de suelo (pestaña «Fondo»)

- **Estudio (generado)**: negro continuo, con horizonte opcional.
- **Conservar el suelo de la foto**: se mantiene el suelo/tapete real sobre el que está la
  miniatura y sólo la pared del fondo pasa a negro. Al no tener que recortar la peana del
  suelo, el recorte es más fácil y la sombra real se conserva. La línea suelo-pared se
  detecta automáticamente (y se puede subir o bajar con «Ajuste de la línea»); si en una
  foto no se distingue, se avisa y se usa el suelo de estudio. El encuadre no sale de la foto.
- **Textura oscura** y **Terreno árido**: suelo generado en la misma perspectiva que la
  peana, más pequeño y desenfocado con la distancia, con su sombra de contacto.

En línea de comandos: `--suelo estudio|original|oscuro|arido` y `--suelo-ajuste`.

## Perfiles de ajustes

En la pestaña **Perfiles** del panel puedes guardar todos los ajustes actuales con un
nombre (p. ej. «Instagram cuadrado con horizonte») y volver a aplicarlos cuando quieras.
Uno de ellos puede marcarse con **Usar al abrir**: sus valores se cargan cada vez que
abres la aplicación. «Valores de fábrica» devuelve todo a los valores originales.

Los perfiles se guardan en `~/.fotos-minis/perfiles.json`, fuera de la carpeta del
programa, así que se conservan al actualizar la aplicación.

Desde la línea de comandos se usa por defecto el mismo perfil «al abrir»; también puedes
elegir otro con `--perfil "Nombre"` y ver la lista con `--perfiles`. Cualquier opción
que escribas (p. ej. `--proporcion 1:1`) prevalece sobre el perfil.

## Apariencia

En la cabecera puedes elegir **Sistema** (sigue el modo claro/oscuro de tu ordenador),
**Diurno** o **Nocturno**. La elección se recuerda para las siguientes aperturas
(`~/.fotos-minis/preferencias.json`).

## Formatos

Acepta JPG, PNG, WebP, TIFF y **HEIC/HEIF** (fotos de iPhone). El perfil de color de la
foto original (p. ej. Display P3) se conserva en el resultado para que los colores de la
pintura no cambien. En la interfaz web el navegador no muestra la miniatura de un HEIC
en el recuadro de entrada, pero se procesa igual y aparece en el comparador.

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
