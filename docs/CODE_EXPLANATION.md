# Explicación técnica del pipeline

Este documento describe en profundidad cómo funciona internamente el
código del proyecto: el flujo de datos entre módulos, qué hace cada
función y por qué se tomaron ciertas decisiones metodológicas. Para
instrucciones de instalación y ejecución rápida, consulta el
[`README.md`](../README.md); aquí el foco es el **funcionamiento interno**.

## Flujo del pipeline

El pipeline transforma una grabación de voz cruda (en cualquier formato
soportado) en una versión digitalizada, limpia y analizada en frecuencia,
pasando por seis etapas secuenciales. Cada etapa consume la salida de la
anterior:

```
 Audio crudo (wav/mp3/m4a/ogg/...)
        │
        ▼
 [1] format_utils.convert_to_wav          (solo si el formato no es .wav)
        │  -> audio en WAV, mismo sample rate original
        ▼
 [2] load_and_inspect.load_audio          -> señal (numpy array) + sample rate
        │
        ├─► inspect_audio                 -> imprime duración, RMS, amplitud, etc.
        └─► plot_waveform                 -> figures/01_raw_waveform.png
        │
        ▼
 [3] preprocess.remove_silence            -> señal sin silencios/ruido + intervalos de voz
        │  -> audio/cleaned_recording.wav
        └─► plot_silence_removal          -> figures/02_silence_removal.png
        │
        ▼
 [4] sampling_quant_code
        ├─► resample_signal               -> señal a 8000 Hz (por defecto)
        ├─► quantize_signal               -> señal cuantizada a 8 bits + SNR
        ├─► encode_signal                 -> códigos PCM binarios + bitrate
        │     -> audio/quantized_recording.wav
        ├─► plot_sampling_quantization    -> figures/03_sampling_quantization.png
        └─► plot_quantization_comparison  -> figures/04_bit_depth_comparison.png
        │
        ▼
 [5] fourier_analysis
        ├─► compute_fft                   -> espectro completo (frecuencia, magnitud, dB)
        ├─► plot_spectrum                 -> figures/05_frequency_spectrum.png
        ├─► compute_spectrogram_dominant_freqs -> frecuencia dominante por frame (STFT)
        ├─► plot_frequency_histogram      -> figures/06_frequency_histogram.png
        ├─► identify_dominant_frequency   -> frecuencia dominante global (dos métodos)
        └─► identify_frequency_range      -> rango de frecuencias con energía significativa
        │
        ▼
 [6] main.py                              -> orquesta [1]-[5] y consolida un resumen final
```

`main.py` es el único módulo que no aporta lógica de procesamiento propia:
importa las funciones de los cinco módulos anteriores y las ejecuta en
orden, reutilizando exactamente el mismo código que ya se probó de forma
individual en cada módulo.

## Módulo por módulo

### `load_and_inspect.py`

**Problema que resuelve**: cargar un archivo de audio arbitrario en memoria
como un arreglo numérico, y dar una primera idea de sus características
básicas antes de procesarlo.

**Funciones principales**:
- `load_audio(path)`: usa `librosa.load(path, sr=None)` para leer el
  archivo sin remuestrearlo (se conserva el sample rate original del
  archivo). Lanza `FileNotFoundError` si la ruta no existe y `RuntimeError`
  si el archivo existe pero no pudo decodificarse.
- `inspect_audio(signal, sample_rate)`: imprime duración, número de
  muestras, sample rate, amplitud máxima/mínima y valor RMS. Es puramente
  informativa, no transforma la señal.
- `plot_waveform(signal, sample_rate, output_path)`: grafica la forma de
  onda en el dominio del tiempo (eje x en segundos, no en muestras) y la
  guarda como PNG.

**Conexión con el resto**: `load_audio` es importado y reutilizado por
`preprocess.py`, `sampling_quant_code.py`, `fourier_analysis.py` y
`main.py` cada vez que necesitan leer un archivo WAV desde disco. Es la
única puerta de entrada de audio a memoria en todo el pipeline.

### `format_utils.py`

**Problema que resuelve**: el audio de entrada puede venir en formatos
comprimidos (OGG, MP3, M4A, ...), pero el resto del pipeline trabaja sobre
WAV sin pérdida. Este módulo normaliza el formato de entrada.

**Función principal**:
- `convert_to_wav(input_path, output_path=None)`: carga el archivo con
  `librosa.load(sr=None)` (que internamente delega en `libsndfile` o
  `audioread`/`ffmpeg` según el formato) y lo vuelve a guardar como WAV con
  `soundfile.write`, preservando el sample rate original. Si el códec no
  puede decodificarse, lanza un `RuntimeError` con instrucciones para
  convertir el archivo manualmente con `ffmpeg`.

**Conexión con el resto**: `preprocess.py` y `main.py` llaman a
`convert_to_wav` como primer paso, únicamente cuando la extensión del
archivo de entrada no es `.wav`. El resultado se guarda en `audio/`, y a
partir de ahí el pipeline continúa igual que si el usuario hubiera
proporcionado un WAV desde el principio.

### `preprocess.py`

**Problema que resuelve**: las grabaciones de voz reales rara vez son
silencio absoluto entre palabras — suelen tener ruido de fondo audible
(ventilador, tráfico, ruido de micrófono, etc.). Cortar por un umbral de
amplitud fijo cercano a cero no funciona: o dicho ruido se cuela como si
fuera voz, o hay que subir tanto el umbral que se recortan partes reales de
la voz. Este módulo estima el ruido de fondo de la propia grabación y ajusta
el umbral de silencio en consecuencia.

**Funciones principales**:
- `estimate_noise_floor(signal, sample_rate, frame_duration_ms=20)`:
  divide la señal en frames de 20 ms, calcula el RMS de cada uno y toma el
  percentil 10 de esos valores como estimación del piso de ruido (se asume
  que los frames más silenciosos de la grabación son mayormente ruido de
  fondo, no voz). Devuelve esa estimación en amplitud lineal y en dB
  relativos al pico de la señal.
- `remove_silence(signal, sample_rate, top_db=None)`: si no se especifica
  `top_db`, lo calcula como el piso de ruido estimado más un margen de 10 dB
  (acotado entre 10 y 90 dB), y usa `librosa.effects.split(signal, top_db=...)`
  para obtener los intervalos no silenciosos. Concatena esos intervalos en
  una señal limpia y devuelve también los intervalos (en muestras) para
  poder visualizarlos.
- `plot_silence_removal(signal, sample_rate, intervals, output_path)`:
  grafica la señal original y sombrea en naranja las regiones detectadas
  como voz, dejando sin sombrear las regiones removidas como silencio/ruido.

**Conexión con el resto**: recibe la señal ya cargada por `load_and_inspect.load_audio`
(o, si el formato de entrada no era WAV, previamente convertida por
`format_utils.convert_to_wav`). Su salida (`clean_signal`) alimenta a
`sampling_quant_code.py`.

### `sampling_quant_code.py`

**Problema que resuelve**: convertir la señal limpia (que sigue siendo
audio de alta resolución) en una representación digital compacta, propia
de un sistema de transmisión/almacenamiento de voz: muestreo a una
frecuencia menor, cuantización de amplitud a un número reducido de niveles,
y codificación binaria de esos niveles.

**Funciones principales**:
- `resample_signal(signal, orig_sr, target_sr)`: remuestrea con
  `librosa.resample`. Ver la sección de decisiones metodológicas para la
  justificación de `target_sr = 8000 Hz`.
- `quantize_signal(signal, num_bits)`: normaliza la señal a `[-1, 1]`,
  la discretiza en `2**num_bits` niveles enteros uniformemente espaciados, y
  la reescala de vuelta a la amplitud original para poder compararla
  visualmente con la señal sin cuantizar. Calcula el **SNR de
  cuantización** (relación entre la potencia de la señal original y la
  potencia del error de cuantización, en dB) como medida de la calidad
  perdida en el proceso.
- `encode_signal(quantization_levels, num_bits, sample_rate=None)`:
  convierte cada nivel entero en su código binario de `num_bits` bits (PCM),
  concatena todos los códigos en un único bitstream, y reporta el tamaño
  total en bits y el bitrate resultante (`sample_rate * num_bits`).
- `plot_sampling_quantization(...)`: grafica un zoom de pocos milisegundos
  de la señal original superpuesta con la cuantizada (como escalones), para
  ver el efecto de la cuantización muestra a muestra.
- `plot_quantization_comparison(...)`: aplica `quantize_signal` con varias
  profundidades de bits (8, 4, 2 por defecto) y grafica el resultado de cada
  una en subplots apilados, con el SNR correspondiente en el título, para
  visualizar la degradación progresiva de la señal.

**Conexión con el resto**: recibe `clean_signal` de `preprocess.py`. Su
salida (`quantized_signal`, a `target_sr` Hz) es la entrada de
`fourier_analysis.py` — el análisis espectral se hace sobre la señal ya
digitalizada, no sobre el audio original de alta resolución, porque es la
señal que efectivamente "sobrevive" al pipeline de transmisión/almacenamiento
simulado.

### `fourier_analysis.py`

**Problema que resuelve**: caracterizar el contenido en frecuencia de la
señal de voz digitalizada — qué frecuencias contiene, cuál domina, y en qué
rango se concentra la energía relevante.

**Funciones principales**:
- `compute_fft(signal, sample_rate)`: calcula la FFT de toda la señal con
  `scipy.fft.rfft` (espectro de un solo lado, válido porque la señal es
  real y su espectro completo sería simétrico) y las frecuencias
  correspondientes con `scipy.fft.rfftfreq`. Devuelve la magnitud lineal y
  su versión en dB, normalizada respecto al pico del espectro (0 dB = pico).
- `plot_spectrum(...)`: grafica esa magnitud en dB vs. frecuencia, limitando
  el eje x a `[0, sample_rate/2]` (el límite de Nyquist).
- `compute_spectrogram_dominant_freqs(signal, sample_rate, frame_length, hop_length)`:
  en lugar de un único espectro para toda la señal, calcula un
  espectrograma vía `librosa.stft` y, para cada frame (columna), identifica
  el bin de frecuencia con mayor magnitud. Los frames cuya energía total es
  despreciable frente al frame más energético (por defecto, menos del 1%)
  se descartan, para no contaminar el análisis con "frecuencias dominantes"
  que en realidad son silencio residual o ruido de piso.
- `plot_frequency_histogram(...)`: histograma de esas frecuencias
  dominantes por frame, marcando con una línea vertical la moda (el bin más
  frecuente).
- `identify_dominant_frequency(...)`: compara dos estimaciones
  independientes de la frecuencia dominante — el pico del espectro completo
  y la moda del histograma por frame — y avisa si difieren
  significativamente (más del 20% relativo), lo cual puede indicar ruido de
  fondo con energía concentrada en otra banda o alta variabilidad de la voz
  entre frames.
- `identify_frequency_range(freqs, magnitude_db, threshold_db=-20)`:
  encuentra la frecuencia mínima y máxima cuya magnitud está por encima de
  `threshold_db` relativo al pico, es decir, el ancho de banda donde se
  concentra la energía significativa de la señal.

**Conexión con el resto**: recibe `quantized_signal` y `target_sr` de
`sampling_quant_code.py`. Es la última etapa de análisis del pipeline; sus
resultados (frecuencia dominante y rango de frecuencias) se incluyen en el
resumen final que imprime `main.py`.

### `main.py`

**Problema que resuelve**: ejecutar el pipeline completo con un solo
comando, sin tener que correr cada módulo por separado ni pasar
manualmente los archivos intermedios de uno a otro.

**Cómo funciona**: importa las funciones de los cinco módulos anteriores
(no reimplementa ninguna) y las ejecuta en el orden descrito en el
diagrama de flujo. Además:
- Si no se indica un archivo de entrada por línea de comandos, busca
  automáticamente el primer archivo de audio dentro de `audio/` que no sea
  uno de los archivos que el propio pipeline genera
  (`cleaned_recording.wav`, `quantized_recording.wav`), para no
  reprocesar por error una salida anterior como si fuera la entrada.
  (`_find_default_input_audio`).
- Guarda los archivos de audio intermedios y finales
  (`audio/cleaned_recording.wav`, `audio/quantized_recording.wav`) en los
  mismos lugares donde los guardarían `preprocess.py` y
  `sampling_quant_code.py` si se ejecutaran de forma individual.
- Al final, imprime un único resumen consolidado con las métricas más
  relevantes de todas las etapas: duración original, porcentaje de audio
  removido como silencio, sample rate original y final, bits de
  cuantización, SNR, bitrate, frecuencia dominante (por los dos métodos) y
  rango de frecuencias detectado.

## Decisiones metodológicas clave

### ¿Por qué 8000 Hz de sample rate?

Por el **teorema de Nyquist**, una señal muestreada a `fs` Hz solo puede
representar sin ambigüedad (sin aliasing) frecuencias de hasta `fs/2` Hz.
La voz humana concentra su inteligibilidad —la frecuencia fundamental y los
primeros formantes— por debajo de aproximadamente 4 kHz. Muestrear a
8000 Hz (2 × 4000 Hz) es, por tanto, el mínimo estándar histórico usado en
telefonía y aplicaciones de voz: suficiente para que la voz siga siendo
inteligible, con un costo de almacenamiento/procesamiento mucho menor que
usar, por ejemplo, 44100 Hz (el estándar de audio musical de alta
fidelidad).

### ¿Por qué 8 bits de cuantización?

8 bits (256 niveles) es la profundidad de referencia clásica en códecs de
voz digital (por ejemplo, PCM de 8 bits tipo µ-law/A-law usados en
telefonía). Ofrece un equilibrio razonable entre calidad audible —el SNR de
cuantización resultante suele rondar los 30 dB para voz normalizada, lo
cual es aceptable para inteligibilidad— y tamaño de datos: reducir a 4 o 2
bits (como se muestra explícitamente en la comparación de
`plot_quantization_comparison`) degrada notablemente el SNR y la calidad
percibida, mientras que subir a 16 bits (el estándar de audio de alta
fidelidad) aporta una calidad que excede lo necesario para el propósito de
este pipeline.

### ¿Por qué un umbral adaptativo de silencio en vez de uno fijo?

Un umbral de amplitud fijo (por ejemplo, "todo lo que esté por debajo de
0.01 es silencio") asume que el silencio es amplitud cercana a cero. En una
grabación real con ruido de fondo audible, eso no es cierto: el "silencio"
entre palabras tiene una amplitud de ruido de fondo distinta de cero, y
puede variar de una grabación a otra (una habitación silenciosa vs. una con
ventilador, tráfico, etc.). Por eso `preprocess.estimate_noise_floor` mide
el ruido de fondo *de la propia grabación* (percentil 10 del RMS por
frames), y `remove_silence` fija el umbral de corte (`top_db`) algunos dB
por encima de ese ruido de fondo específico, en lugar de usar un valor
arbitrario que funcionaría bien en una grabación y mal en otra.

### ¿Cómo se calculan la frecuencia dominante y el rango de frecuencias?

La frecuencia dominante se calcula de **dos formas independientes** que se
contrastan entre sí:
1. El pico del espectro de toda la señal (un único FFT global): útil como
   resumen general, pero puede diluirse si la frecuencia fundamental de la
   voz varía a lo largo de la grabación.
2. La moda de las frecuencias dominantes por frame (vía STFT): calcula la
   frecuencia dominante en ventanas cortas de tiempo y toma la que más se
   repite, lo cual es más robusto ante variaciones temporales de la voz,
   pero puede verse afectado por frames individuales ruidosos (por eso se
   descartan los frames de energía despreciable antes de calcular la moda).

Cuando ambos métodos coinciden aproximadamente, hay buena confianza en el
resultado; cuando difieren significativamente, suele ser señal de ruido de
fondo con energía concentrada en otra banda, o de una voz con mucha
variación tonal.

El **rango de frecuencias** se define como la banda donde la magnitud del
espectro se mantiene por encima de un umbral relativo al pico (-20 dB por
defecto): fuera de ese rango, la energía se considera despreciable frente
al contenido principal de la señal. Como el pipeline remuestrea a 8000 Hz,
este análisis está inherentemente limitado a frecuencias por debajo de
4000 Hz (Nyquist) — suficiente para la frecuencia fundamental de la voz,
pero recorta los armónicos superiores de los formantes que, en una voz
natural sin remuestrear, pueden extenderse más allá de 4 kHz.
