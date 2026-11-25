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

### Como aplicación de Mac (recomendado)

Haz doble clic en **`Instalar app en Aplicaciones.command`** (la primera vez, clic
derecho → Abrir). Prepara lo necesario y crea **Fotos Minis.app** en Aplicaciones, con su
icono: aparece en Launchpad y puedes arrastrarla al Dock. Al abrirla se muestra en **su
propia ventana**, sin navegador ni ventana de Terminal; al cerrar la ventana (o con
«Cerrar aplicación») se detiene todo.

- La app usa esta carpeta del programa: si la mueves, vuelve a ejecutar el instalador.
- Tras actualizar el programa no hace falta reinstalar la app (sólo si cambia este apartado).
- Registro de arranque y errores: `~/.fotos-minis/app.log`.
- Si la ventana propia no pudiera abrirse, la app se abre en el navegador.
- **Icono propio**: al ejecutar el instalador, arrastra a la ventana de Terminal una imagen
  (JPG, PNG o HEIC; mejor un primer plano de una sola miniatura) y pulsa Intro. Se recorta en
  cuadrado con las esquinas de los iconos de macOS y se recuerda en las siguientes
  instalaciones; escribe `original` para volver al icono por defecto (`recursos/icono.png`; para cambiar el de por defecto basta con sustituir ese fichero por otro PNG de 1024×1024). Si el Dock sigue
  mostrando el anterior, quita la app del Dock y vuelve a añadirla.

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

## Carrusel de Instagram (pestaña «Encuadre»)

Elige **Auto** o de **2 a 6 partes**. En Auto, para cada foto se usa el número de partes
con el que las minis ocupan más área (en un lote, cada foto el suyo). Después: la imagen se encuadra en proporción N:1 (N cuadrados seguidos),
lo más ajustada posible a las minis con el margen indicado y sin reescalarlas, y se corta en
N imágenes 1:1. Publicadas juntas como carrusel, al deslizar se ven como una sola imagen
continua (los cortes coinciden píxel a píxel).

Se guardan la imagen completa antes de cortar (`<nombre>_estudio.png`) y las partes
(`<nombre>_carrusel_1de3.png`, …); en otra prueba con la misma foto se sustituyen todas. La
pestaña **Carrusel** del visor las muestra, la descarga es un ZIP con todo, y la app indica
cuántas partes aprovechan mejor el espacio en esa foto. Línea de comandos: `--carrusel 3` o `--carrusel auto`.

## Fotos individuales de cada figura (pestaña «Encuadre»)

Marca **Fotos individuales de cada figura**: si en la foto hay varias minis, además de la
foto de grupo se guarda una por figura (`<nombre>_figura1.png`, … de izquierda a derecha),
**centrada y entera** (brazos levantados y armas incluidos) con la proporción objetivo (la
de «Proporción», o 1:1 si el carrusel está activo).

- Cada figura es todo lo que está unido a su peana; si varias se solapan, se separan por la
  divisoria entre ellas (la de delante tiene preferencia).
- Si el encuadre centrado cabe en la foto de grupo y no asoma ninguna otra figura, se
  **recorta** de ella. Si no, esa figura se **vuelve a componer sola con un fondo nuevo**
  (mismo fondo, suelo y luz; se reutiliza su máscara, no se segmenta otra vez).
- Las figuras casi ocultas tras otras no pueden mostrarse enteras: se omiten y se avisa.
- **Super-resolución (IA)** opcional, ×2 o ×4 (máximo 4096 px): amplía las fotos individuales
  con Real-ESRGAN para que los recortes pequeños no queden borrosos. Tarda unos segundos por
  figura en CPU. Ojo: al ser IA, reinterpreta los detalles más finos (la figura ya no es
  píxel a píxel la original); por eso es opcional. Terminal: `--superres 2`.
- Con una sola figura no se genera nada. Se ven en la pestaña **Individuales** (indicando si
  son recorte o fondo nuevo), van en el ZIP y en otra prueba se sustituyen. Terminal:
  `--individuales`.

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
