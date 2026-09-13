# Audio Processing Project

Proyecto para el curso **Python for Research**: un pipeline de procesamiento
de señales de audio de voz construido paso a paso.

## Objetivo

Cargar una grabación de voz ya existente y aplicarle, en pasos sucesivos,
distintas etapas de un pipeline clásico de procesamiento digital de señales
(inspección, filtrado de silencios, muestreo, cuantización, codificación,
análisis en frecuencia con FFT, etc.). Este repositorio contiene el **primer
paso**: carga e inspección básica del audio.

## Audio de entrada

Este proyecto **no graba audio**. Debes colocar tu propio archivo de voz ya
grabado dentro de la carpeta `audio/`.

Formatos soportados (gracias a `librosa`/`soundfile`): `.wav`, `.mp3`, `.m4a`,
entre otros formatos comunes.

Por defecto, el script busca `audio/raw_recording.wav`. Si tu archivo tiene
otro nombre o formato, puedes indicarlo por línea de comandos (ver abajo).

## Estructura del proyecto

```
audio-processing-project/
├── audio/              # Coloca aquí tu archivo de audio de voz
├── figures/            # Gráficas generadas por el pipeline
├── src/                # Código fuente del pipeline
│   ├── load_and_inspect.py
│   ├── format_utils.py
│   ├── preprocess.py
│   ├── sampling_quant_code.py
│   ├── fourier_analysis.py
│   └── main.py         # Orquestador end-to-end
├── docs/
│   └── CODE_EXPLANATION.md  # Explicación técnica interna del pipeline
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso: `load_and_inspect.py`

Este módulo carga el audio, imprime un resumen de sus características y
genera una gráfica de la forma de onda en el dominio del tiempo.

Usando el archivo por defecto (`audio/raw_recording.wav`):

```bash
python src/load_and_inspect.py
```

Especificando otro archivo:

```bash
python src/load_and_inspect.py audio/mi_grabacion.mp3
```

La gráfica resultante se guarda en `figures/01_raw_waveform.png`.

## Soporte de múltiples formatos y filtrado adaptativo de silencios

El pipeline admite archivos de entrada en varios formatos comunes (WAV, OGG,
MP3, M4A, ...). Cuando el archivo no está en WAV, `src/format_utils.py`
lo convierte automáticamente a WAV (vía `librosa`/`soundfile`) antes de
procesarlo; si el códec no puede decodificarse, se indica cómo convertirlo
manualmente con `ffmpeg`.

Como las grabaciones de voz suelen tener ruido de fondo audible, el filtrado
de silencios (`src/preprocess.py`) **no usa un umbral fijo de amplitud**.
En su lugar, estima el piso de ruido de la grabación (percentil bajo del RMS
por frames cortos) y calcula un umbral adaptativo (`top_db`) algunos dB por
encima de ese ruido de fondo, usando `librosa.effects.split` para separar
los tramos de voz de los de silencio/ruido. El resultado se guarda en
`audio/cleaned_recording.wav` y se genera una gráfica
(`figures/02_silence_removal.png`) que sombrea las regiones detectadas como
voz, para verificar visualmente que el umbral fue razonable.

Uso:

```bash
python src/preprocess.py audio/mi_grabacion.ogg
```

## Muestreo, cuantización y codificación

`src/sampling_quant_code.py` toma el audio limpio (`audio/cleaned_recording.wav`)
y aplica las tres etapas clásicas de digitalización de una señal de voz:

- **Muestreo**: remuestrea la señal a `target_sr` (por defecto **8000 Hz**).
  Por el teorema de Nyquist, la frecuencia de muestreo debe ser al menos el
  doble de la frecuencia máxima de interés de la señal. La voz humana
  concentra su inteligibilidad por debajo de ~4 kHz, por lo que 8000 Hz
  (2 × 4000 Hz) es el estándar típico usado en telefonía y aplicaciones de voz.
- **Cuantización**: discretiza la amplitud en `2**num_bits` niveles uniformes,
  usando por defecto **8 bits** (256 niveles), la profundidad de referencia
  clásica en códecs de voz (ej. PCM de 8 bits tipo µ-law/A-law en telefonía),
  que ofrece un buen equilibrio entre calidad audible y tamaño de datos.
- **Codificación**: convierte cada nivel cuantizado a su código binario PCM
  de `num_bits` bits y reporta el bitrate resultante (`sample_rate * num_bits`).

El script guarda la señal cuantizada reconstruida en
`audio/quantized_recording.wav` y genera dos gráficas: un zoom de la señal
original vs. cuantizada (`figures/03_sampling_quantization.png`) y una
comparación del efecto de distintas profundidades de bits
(`figures/04_bit_depth_comparison.png`).

Uso:

```bash
python src/sampling_quant_code.py
python src/sampling_quant_code.py audio/cleaned_recording.wav --target-sr 16000 --num-bits 4
```

## Ejecución desde cualquier carpeta

Los scripts funcionan sin importar desde qué directorio se ejecuten (por
ejemplo, si la terminal integrada de VS Code se abre en otra carpeta). Cada
módulo calcula su propia raíz de proyecto (`PROJECT_ROOT`) a partir de la
ubicación del archivo `.py`, y todas las rutas por defecto de entrada/salida
(`audio/*.wav`, `figures/*.png`) se construyen relativas a esa raíz en vez
de depender del directorio de trabajo actual. Si le pasas una ruta relativa
propia por línea de comandos, primero se intenta resolver contra el
directorio actual y, si no existe ahí, contra la raíz del proyecto; si no se
encuentra en ninguna de las dos, el error muestra ambas rutas absolutas
probadas.

## Análisis en frecuencia (FFT)

`src/fourier_analysis.py` toma el audio digitalizado (por defecto
`audio/quantized_recording.wav`) y analiza su contenido espectral:

- **Espectro completo**: calcula la FFT de toda la señal con
  `scipy.fft.rfft` (espectro de un solo lado, ya que la señal es real) y
  grafica la magnitud en dB vs. frecuencia (`figures/05_frequency_spectrum.png`).
- **Frecuencia dominante**: se identifica de dos formas y se comparan entre
  sí: (a) el pico del espectro completo, y (b) la moda de las frecuencias
  dominantes por frame calculadas vía STFT (`librosa.stft`), descartando
  frames de energía despreciable (silencio residual). La distribución de
  frecuencias dominantes por frame se grafica como histograma
  (`figures/06_frequency_histogram.png`), marcando la moda con una línea
  vertical.
- **Rango de frecuencias efectivo**: se determina dónde la magnitud del
  espectro supera un umbral relativo al pico (por defecto -20 dB), es decir,
  dónde se concentra la energía significativa de la voz.

**Importante sobre el sample_rate**: como el pipeline remuestrea a 8000 Hz,
por Nyquist el análisis solo puede observar frecuencias por debajo de
4000 Hz. Esto es suficiente para capturar la frecuencia fundamental de la
voz humana (típicamente 85-255 Hz), pero recorta los armónicos superiores
de los formantes, que en una voz natural pueden extenderse más allá de
4 kHz.

Uso:

```bash
python src/fourier_analysis.py
python src/fourier_analysis.py audio/quantized_recording.wav
```

## Pipeline completo de un solo comando

`src/main.py` ejecuta todo el pipeline de principio a fin (conversión de
formato, inspección, filtrado de silencios, muestreo/cuantización/codificación
y análisis en frecuencia) reutilizando las funciones de los demás módulos,
sin reimplementar lógica. Si no se indica un archivo de entrada, busca
automáticamente el primer archivo de audio en `audio/` que no sea uno de los
generados por el propio pipeline (`cleaned_recording.wav`,
`quantized_recording.wav`). Al finalizar, genera las seis figuras (`01` a
`06` en `figures/`) e imprime un único resumen consolidado con las métricas
más relevantes de todas las etapas.

Uso:

```bash
python src/main.py
python src/main.py audio/mi_grabacion.ogg --target-sr 16000 --num-bits 4 --threshold-db -25
```

## Documentación técnica adicional

Para una explicación más profunda de cómo funciona el código internamente
(arquitectura, rol de cada módulo y justificación de las decisiones
metodológicas), consulta [`docs/CODE_EXPLANATION.md`](docs/CODE_EXPLANATION.md).

## Próximos pasos

El pipeline base (carga, filtrado de silencios, muestreo/cuantización/codificación
y análisis en frecuencia) está completo, con un orquestador end-to-end.
Posibles extensiones futuras: filtros pasa-banda, extracción de
características (MFCC), o comparación de espectros entre distintas
grabaciones.
