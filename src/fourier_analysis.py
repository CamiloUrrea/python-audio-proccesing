"""
fourier_analysis.py

Módulo de análisis en frecuencia del pipeline de procesamiento de audio.

Toma la señal ya remuestreada/cuantizada (ver sampling_quant_code.py) y
analiza su contenido espectral: calcula la Transformada de Fourier (FFT) de
toda la señal, estima cómo varía la frecuencia dominante a lo largo del
tiempo (vía STFT) y determina el rango de frecuencias donde se concentra la
energía de la voz.
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq

from load_and_inspect import load_audio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_AUDIO_PATH = PROJECT_ROOT / "audio" / "quantized_recording.wav"
DEFAULT_SPECTRUM_PLOT_PATH = PROJECT_ROOT / "figures" / "05_frequency_spectrum.png"
DEFAULT_HISTOGRAM_PLOT_PATH = PROJECT_ROOT / "figures" / "06_frequency_histogram.png"

DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_MIN_ENERGY_RATIO = 0.01
DEFAULT_THRESHOLD_DB = -20
DEFAULT_HISTOGRAM_BINS = 50


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


def compute_fft(signal, sample_rate):
    """
    Calcula la Transformada de Fourier de la señal completa.

    Se usa scipy.fft.rfft (espectro de un solo lado) porque la señal de
    audio es real: para señales reales, el espectro completo es simétrico
    respecto al eje de frecuencias, por lo que la mitad "negativa" no aporta
    información nueva.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio (real).
    sample_rate : int
        Frecuencia de muestreo de la señal, en Hz.

    Returns
    -------
    freqs : numpy.ndarray
        Frecuencias correspondientes a cada componente, en Hz (de 0 a
        sample_rate/2, el límite de Nyquist).
    magnitude : numpy.ndarray
        Magnitud lineal de cada componente de frecuencia.
    magnitude_db : numpy.ndarray
        Magnitud en escala logarítmica (dB), normalizada respecto al valor
        máximo (0 dB corresponde al pico del espectro).
    """
    fft_result = rfft(signal)
    freqs = rfftfreq(len(signal), d=1 / sample_rate)

    magnitude = np.abs(fft_result)

    epsilon = 1e-12
    peak_magnitude = np.max(magnitude)
    magnitude_db = 20 * np.log10((magnitude + epsilon) / (peak_magnitude + epsilon))

    return freqs, magnitude, magnitude_db


def plot_spectrum(freqs, magnitude_db, sample_rate, output_path=DEFAULT_SPECTRUM_PLOT_PATH):
    """
    Grafica el espectro de magnitud (dB) en función de la frecuencia.

    Parameters
    ----------
    freqs : numpy.ndarray
        Frecuencias, en Hz (ver compute_fft).
    magnitude_db : numpy.ndarray
        Magnitud en dB, normalizada al pico (ver compute_fft).
    sample_rate : int
        Frecuencia de muestreo usada, en Hz (define el límite de Nyquist
        del eje x).
    output_path : str or Path, optional
        Ruta donde se guardará la gráfica generada.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude_db, linewidth=0.7)
    plt.xlim(0, sample_rate / 2)
    plt.title("Espectro de magnitud (FFT)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()

    print(f"Gráfica guardada en: {output_path}")


def compute_spectrogram_dominant_freqs(
    signal,
    sample_rate,
    frame_length=DEFAULT_FRAME_LENGTH,
    hop_length=DEFAULT_HOP_LENGTH,
    min_energy_ratio=DEFAULT_MIN_ENERGY_RATIO,
):
    """
    Calcula la frecuencia dominante de cada frame de la señal, vía STFT.

    Para cada frame (columna del espectrograma) se busca el bin de
    frecuencia con mayor magnitud. Los frames cuya energía total es
    despreciable frente al máximo (ej. silencio residual entre palabras) se
    descartan usando un umbral relativo, para que no distorsionen el
    análisis con "frecuencias dominantes" que en realidad son solo ruido.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio.
    sample_rate : int
        Frecuencia de muestreo, en Hz.
    frame_length : int, optional
        Tamaño de la ventana FFT usada por el STFT (n_fft).
    hop_length : int, optional
        Desplazamiento entre frames consecutivos.
    min_energy_ratio : float, optional
        Fracción mínima de la energía máxima de un frame (respecto al frame
        más energético) para considerarlo válido.

    Returns
    -------
    numpy.ndarray
        Frecuencias dominantes (Hz), una por cada frame válido.
    """
    stft_matrix = librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)
    magnitude_spectrogram = np.abs(stft_matrix)
    freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)

    frame_energy = np.sum(magnitude_spectrogram ** 2, axis=0)
    energy_threshold = min_energy_ratio * np.max(frame_energy)
    valid_frames_mask = frame_energy > energy_threshold

    dominant_bin_indices = np.argmax(magnitude_spectrogram[:, valid_frames_mask], axis=0)
    dominant_freqs_per_frame = freq_bins[dominant_bin_indices]

    total_frames = magnitude_spectrogram.shape[1]
    valid_frames = int(np.sum(valid_frames_mask))
    print(
        f"Frames con energía significativa: {valid_frames}/{total_frames} "
        f"(umbral: {min_energy_ratio * 100:.0f}% de la energía máxima por frame)"
    )

    return dominant_freqs_per_frame


def _compute_mode_frequency(dominant_freqs_per_frame, bins=DEFAULT_HISTOGRAM_BINS):
    """
    Calcula la moda (bin más frecuente) de un histograma de frecuencias.

    Parameters
    ----------
    dominant_freqs_per_frame : numpy.ndarray
        Frecuencias dominantes por frame (ver compute_spectrogram_dominant_freqs).
    bins : int, optional
        Número de bins del histograma.

    Returns
    -------
    float
        Frecuencia (centro del bin) con mayor número de frames.
    """
    counts, bin_edges = np.histogram(dominant_freqs_per_frame, bins=bins)
    mode_bin_index = np.argmax(counts)
    mode_freq = (bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2
    return mode_freq


def plot_frequency_histogram(
    dominant_freqs_per_frame,
    output_path=DEFAULT_HISTOGRAM_PLOT_PATH,
    bins=DEFAULT_HISTOGRAM_BINS,
):
    """
    Grafica un histograma de las frecuencias dominantes por frame.

    Parameters
    ----------
    dominant_freqs_per_frame : numpy.ndarray
        Frecuencias dominantes por frame (ver compute_spectrogram_dominant_freqs).
    output_path : str or Path, optional
        Ruta donde se guardará la gráfica generada.
    bins : int, optional
        Número de bins del histograma.
    """
    mode_freq = _compute_mode_frequency(dominant_freqs_per_frame, bins=bins)

    plt.figure(figsize=(10, 4))
    plt.hist(dominant_freqs_per_frame, bins=bins, color="steelblue", edgecolor="white")
    plt.axvline(
        mode_freq, color="red", linestyle="--", label=f"Moda: {mode_freq:.1f} Hz"
    )
    plt.title("Histograma de frecuencias dominantes por frame")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Número de frames")
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()

    print(f"Gráfica guardada en: {output_path}")


def identify_dominant_frequency(freqs, magnitude_db, dominant_freqs_per_frame):
    """
    Determina la frecuencia dominante global de la señal, de dos formas.

    (a) El pico del espectro de toda la señal (mayor magnitud en compute_fft).
    (b) La moda del histograma de frecuencias dominantes por frame (la
        frecuencia que más veces "gana" a lo largo del tiempo).

    Ambas suelen coincidir aproximadamente para voz con poco ruido, pero
    pueden diferir si hay ruido de fondo con energía concentrada en otra
    banda, o si la voz varía mucho de frecuencia entre frames.

    Parameters
    ----------
    freqs : numpy.ndarray
        Frecuencias del espectro completo (ver compute_fft).
    magnitude_db : numpy.ndarray
        Magnitud en dB del espectro completo (ver compute_fft).
    dominant_freqs_per_frame : numpy.ndarray
        Frecuencias dominantes por frame (ver compute_spectrogram_dominant_freqs).

    Returns
    -------
    peak_spectrum_freq : float
        Frecuencia del pico del espectro completo, en Hz.
    mode_histogram_freq : float
        Moda del histograma de frecuencias dominantes por frame, en Hz.
    """
    peak_spectrum_freq = freqs[np.argmax(magnitude_db)]
    mode_histogram_freq = _compute_mode_frequency(dominant_freqs_per_frame)

    print("Frecuencia dominante")
    print("---------------------")
    print(f"(a) Pico del espectro completo:      {peak_spectrum_freq:.1f} Hz")
    print(f"(b) Moda de frecuencias por frame:    {mode_histogram_freq:.1f} Hz")

    reference = max(peak_spectrum_freq, mode_histogram_freq, 1.0)
    relative_difference = abs(peak_spectrum_freq - mode_histogram_freq) / reference

    if relative_difference > 0.2:
        print(
            "Nota: los dos valores difieren significativamente. Esto puede "
            "deberse a ruido de fondo con energía concentrada en otra banda, "
            "o a que la frecuencia de la voz varía bastante entre frames."
        )

    return peak_spectrum_freq, mode_histogram_freq


def identify_frequency_range(freqs, magnitude_db, threshold_db=DEFAULT_THRESHOLD_DB):
    """
    Determina el rango de frecuencias con energía significativa.

    Se considera "significativa" toda componente cuya magnitud esté por
    encima de threshold_db relativo al pico del espectro (0 dB).

    Parameters
    ----------
    freqs : numpy.ndarray
        Frecuencias del espectro completo (ver compute_fft).
    magnitude_db : numpy.ndarray
        Magnitud en dB del espectro completo, normalizada al pico (ver
        compute_fft).
    threshold_db : float, optional
        Umbral, en dB relativos al pico, por debajo del cual una componente
        se considera energéticamente despreciable.

    Returns
    -------
    freq_min : float
        Frecuencia mínima del rango con energía significativa, en Hz.
    freq_max : float
        Frecuencia máxima del rango con energía significativa, en Hz.
    """
    significant_mask = magnitude_db >= threshold_db

    if not np.any(significant_mask):
        freq_min, freq_max = 0.0, 0.0
    else:
        freq_min = float(freqs[significant_mask].min())
        freq_max = float(freqs[significant_mask].max())

    nyquist_limit = freqs.max()

    print("Rango de frecuencias con energía significativa")
    print("------------------------------------------------")
    print(f"Umbral usado:  {threshold_db} dB relativo al pico")
    print(f"Rango:         {freq_min:.1f} Hz - {freq_max:.1f} Hz")
    print(
        f"Nota: el análisis está limitado a frecuencias por debajo de "
        f"{nyquist_limit:.0f} Hz (límite de Nyquist para el sample_rate usado); "
        "cualquier contenido por encima de ese límite no puede observarse."
    )

    return freq_min, freq_max


def main():
    parser = argparse.ArgumentParser(
        description="Analiza el contenido en frecuencia de un archivo de audio de voz."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(DEFAULT_AUDIO_PATH),
        help=f"Ruta al audio a analizar (default: {DEFAULT_AUDIO_PATH})",
    )
    args = parser.parse_args()

    resolved_path = _resolve_input_path(args.audio_path)

    signal, sample_rate = load_audio(resolved_path)

    freqs, _, magnitude_db = compute_fft(signal, sample_rate)
    plot_spectrum(freqs, magnitude_db, sample_rate)

    dominant_freqs_per_frame = compute_spectrogram_dominant_freqs(signal, sample_rate)
    plot_frequency_histogram(dominant_freqs_per_frame)

    peak_freq, mode_freq = identify_dominant_frequency(
        freqs, magnitude_db, dominant_freqs_per_frame
    )
    freq_min, freq_max = identify_frequency_range(freqs, magnitude_db)

    print("\nResumen final")
    print("-------------")
    print(f"Sample rate usado:               {sample_rate} Hz")
    print(f"Frecuencia dominante (pico FFT):  {peak_freq:.1f} Hz")
    print(f"Frecuencia dominante (moda):      {mode_freq:.1f} Hz")
    print(f"Rango de frecuencias detectado:   {freq_min:.1f} Hz - {freq_max:.1f} Hz")


if __name__ == "__main__":
    main()
