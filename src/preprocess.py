"""
preprocess.py

Módulo de filtrado de silencios del pipeline de procesamiento de audio.

El audio de voz de entrada suele contener ruido de fondo audible, por lo que
un umbral de silencio fijo (ej. "amplitud cercana a cero") no funciona bien:
o bien deja pasar el ruido de fondo como si fuera voz, o bien recorta partes
de la voz si el umbral se sube demasiado. Este módulo estima el piso de
ruido de la grabación y calcula, a partir de él, un umbral adaptativo
(top_db) para separar voz de silencio/ruido usando librosa.effects.split.
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import format_utils
from load_and_inspect import load_audio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_AUDIO_PATH = PROJECT_ROOT / "audio" / "raw_recording.wav"
DEFAULT_CLEANED_AUDIO_PATH = PROJECT_ROOT / "audio" / "cleaned_recording.wav"
DEFAULT_SILENCE_PLOT_PATH = PROJECT_ROOT / "figures" / "02_silence_removal.png"

DEFAULT_MARGIN_DB = 10
MIN_TOP_DB = 10
MAX_TOP_DB = 90


def _resolve_input_path(path_str):
    """
    Resuelve una ruta de entrada indicada por el usuario (típicamente vía
    línea de comandos).

    Primero intenta resolverla tal cual contra el directorio de trabajo
    actual (comportamiento normal). Si el archivo no existe ahí, intenta
    resolverla también relativa a la raíz del proyecto (PROJECT_ROOT), para
    que los valores relativos por defecto sigan funcionando aunque el
    script se ejecute desde otra carpeta (ej. la terminal integrada de
    VS Code abierta en un subdirectorio distinto).

    Parameters
    ----------
    path_str : str or Path
        Ruta indicada por el usuario (relativa o absoluta).

    Returns
    -------
    Path
        Ruta resuelta que existe en disco.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe ni relativo al cwd ni relativo a
        PROJECT_ROOT. El mensaje incluye ambas rutas absolutas probadas.
    """
    cwd_path = Path(path_str).resolve()
    if cwd_path.is_file():
        return cwd_path

    project_root_path = (PROJECT_ROOT / path_str).resolve()
    if project_root_path.is_file():
        return project_root_path

    raise FileNotFoundError(
        f"No se encontró el archivo de audio '{path_str}'. Se intentó:\n"
        f"  - Relativo al directorio actual: {cwd_path}\n"
        f"  - Relativo a la raíz del proyecto: {project_root_path}"
    )


def estimate_noise_floor(signal, sample_rate, frame_duration_ms=20):
    """
    Estima el piso de ruido de fondo de la señal.

    Divide la señal en frames cortos, calcula el RMS de cada uno y toma un
    percentil bajo (percentil 10) de esos valores como estimación del nivel
    de ruido de fondo (asumiendo que la mayoría de los frames más silenciosos
    corresponden a ruido de fondo y no a voz).

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio.
    sample_rate : int
        Frecuencia de muestreo, en Hz.
    frame_duration_ms : float, optional
        Duración de cada frame de análisis, en milisegundos.

    Returns
    -------
    noise_floor_rms : float
        Estimación del piso de ruido en amplitud RMS (escala lineal).
    noise_floor_db : float
        La misma estimación expresada en dB, relativa al pico de amplitud
        de la señal (misma referencia que usa librosa.effects.split).
    """
    frame_length = max(int(sample_rate * frame_duration_ms / 1000), 1)

    rms_per_frame = librosa.feature.rms(
        y=signal, frame_length=frame_length, hop_length=frame_length
    )[0]

    noise_floor_rms = np.percentile(rms_per_frame, 10)

    peak_amplitude = np.max(np.abs(signal))
    noise_floor_db = librosa.amplitude_to_db(
        np.array([noise_floor_rms]), ref=peak_amplitude
    )[0]

    return noise_floor_rms, noise_floor_db


def remove_silence(signal, sample_rate, top_db=None):
    """
    Remueve los silencios (y el ruido de fondo) de una señal de audio.

    Si no se especifica top_db, se calcula automáticamente a partir del piso
    de ruido detectado con estimate_noise_floor: el umbral se fija algunos
    dB por encima del ruido de fondo, en vez de usar un valor fijo
    arbitrario, para que el filtrado se adapte al nivel de ruido real de
    la grabación.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio.
    sample_rate : int
        Frecuencia de muestreo, en Hz.
    top_db : float, optional
        Umbral, en dB por debajo del pico de la señal, a partir del cual un
        frame se considera silencio (ver librosa.effects.split). Si es None,
        se calcula automáticamente.

    Returns
    -------
    clean_signal : numpy.ndarray
        Señal resultante tras concatenar únicamente los intervalos con voz.
    intervals : numpy.ndarray
        Arreglo de forma (n_intervalos, 2) con los índices [inicio, fin)
        (en muestras) de cada intervalo no silencioso detectado.
    top_db : float
        El valor de top_db efectivamente utilizado.
    """
    if top_db is None:
        _, noise_floor_db = estimate_noise_floor(signal, sample_rate)
        top_db = -(noise_floor_db + DEFAULT_MARGIN_DB)
        top_db = float(np.clip(top_db, MIN_TOP_DB, MAX_TOP_DB))

    intervals = librosa.effects.split(signal, top_db=top_db)

    if len(intervals) > 0:
        clean_signal = np.concatenate([signal[start:end] for start, end in intervals])
    else:
        clean_signal = signal

    original_duration = len(signal) / sample_rate
    clean_duration = len(clean_signal) / sample_rate
    percent_removed = (1 - clean_duration / original_duration) * 100 if original_duration > 0 else 0.0

    print("Filtrado de silencios")
    print("----------------------")
    print(f"top_db utilizado:          {top_db:.2f} dB")
    print(f"Duración original:         {original_duration:.3f} s")
    print(f"Duración tras remover:     {clean_duration:.3f} s")
    print(f"Porcentaje de audio removido: {percent_removed:.1f}%")

    return clean_signal, intervals, top_db


def plot_silence_removal(signal, sample_rate, intervals, output_path=DEFAULT_SILENCE_PLOT_PATH):
    """
    Grafica la señal original y sombrea las regiones detectadas como voz,
    para verificar visualmente que el umbral de silencio fue razonable
    dado el ruido de fondo presente en la grabación.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio original (sin recortar).
    sample_rate : int
        Frecuencia de muestreo, en Hz.
    intervals : numpy.ndarray
        Intervalos [inicio, fin) en muestras, tal como los devuelve
        remove_silence / librosa.effects.split.
    output_path : str or Path, optional
        Ruta donde se guardará la gráfica generada. Por defecto, dentro de
        la carpeta 'figures/' de la raíz del proyecto.
    """
    time_axis = np.arange(len(signal)) / sample_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal, linewidth=0.5, color="steelblue")

    for i, (start, end) in enumerate(intervals):
        plt.axvspan(
            start / sample_rate,
            end / sample_rate,
            color="orange",
            alpha=0.3,
            label="Voz detectada" if i == 0 else None,
        )

    plt.title("Detección de voz vs. silencio/ruido de fondo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    if len(intervals) > 0:
        plt.legend(loc="upper right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()

    print(f"Gráfica guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filtra silencios y ruido de fondo de un archivo de audio de voz."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(DEFAULT_AUDIO_PATH),
        help=f"Ruta al archivo de audio a procesar (default: {DEFAULT_AUDIO_PATH})",
    )
    args = parser.parse_args()

    resolved_path = _resolve_input_path(args.audio_path)

    if resolved_path.suffix.lower() != ".wav":
        wav_path = format_utils.convert_to_wav(resolved_path)
        print(f"Archivo convertido a WAV: {wav_path}")
    else:
        wav_path = resolved_path

    signal, sample_rate = load_audio(wav_path)

    clean_signal, intervals, _ = remove_silence(signal, sample_rate)

    output_audio_path = DEFAULT_CLEANED_AUDIO_PATH
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio_path, clean_signal, sample_rate)
    print(f"Señal limpia guardada en: {output_audio_path}")

    plot_silence_removal(signal, sample_rate, intervals)


if __name__ == "__main__":
    main()
