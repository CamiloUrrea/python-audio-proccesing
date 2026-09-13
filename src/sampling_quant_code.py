"""
sampling_quant_code.py

Módulo de muestreo, cuantización y codificación del pipeline de
procesamiento de audio.

Toma la señal ya limpia (sin silencios, ver preprocess.py) y aplica las
etapas clásicas de digitalización de una señal analógica de voz:
1. Muestreo (remuestreo a una frecuencia adecuada para voz).
2. Cuantización (discretización de la amplitud en 2**num_bits niveles).
3. Codificación (representación binaria PCM de cada nivel).
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from load_and_inspect import load_audio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_AUDIO_PATH = PROJECT_ROOT / "audio" / "cleaned_recording.wav"
DEFAULT_QUANTIZED_AUDIO_PATH = PROJECT_ROOT / "audio" / "quantized_recording.wav"
DEFAULT_SAMPLING_PLOT_PATH = PROJECT_ROOT / "figures" / "03_sampling_quantization.png"
DEFAULT_BIT_DEPTH_PLOT_PATH = PROJECT_ROOT / "figures" / "04_bit_depth_comparison.png"

DEFAULT_TARGET_SR = 8000
DEFAULT_NUM_BITS = 8


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


def resample_signal(signal, orig_sr, target_sr):
    """
    Remuestrea una señal de orig_sr a target_sr usando librosa.

    Nota (Teorema de Nyquist): para poder reconstruir sin ambigüedad (aliasing)
    las frecuencias de interés presentes en la señal, la frecuencia de muestreo
    debe ser al menos el doble de la frecuencia máxima de esa señal. La voz
    humana concentra la mayor parte de su energía e inteligibilidad por debajo
    de ~4 kHz, por lo que un target_sr de 8000 Hz (2 x 4000 Hz) es el mínimo
    típico usado en telefonía y aplicaciones de voz.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio original.
    orig_sr : int
        Frecuencia de muestreo original de la señal, en Hz.
    target_sr : int
        Frecuencia de muestreo deseada, en Hz.

    Returns
    -------
    numpy.ndarray
        Señal remuestreada a target_sr.
    """
    print(f"Remuestreando: {orig_sr} Hz -> {target_sr} Hz")

    resampled = librosa.resample(y=signal, orig_sr=orig_sr, target_sr=target_sr)
    return resampled


def quantize_signal(signal, num_bits):
    """
    Cuantiza una señal a 2**num_bits niveles uniformes.

    La señal se normaliza primero al rango [-1, 1] usando su valor máximo
    absoluto, se discretiza en niveles enteros uniformemente espaciados, y
    luego se reescala de vuelta al rango original de amplitud para poder
    compararla directamente con la señal original.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio a cuantizar.
    num_bits : int
        Número de bits de la cuantización (2**num_bits niveles).

    Returns
    -------
    quantized_signal : numpy.ndarray
        Señal cuantizada, en la misma escala de amplitud que la señal original.
    level_indices : numpy.ndarray
        Índices enteros de cuantización (valores entre 0 y 2**num_bits - 1).
    quantization_error : numpy.ndarray
        Señal original menos señal cuantizada.
    snr_db : float
        Relación señal a ruido de cuantización, en dB.
    """
    num_levels = 2 ** num_bits

    peak_amplitude = np.max(np.abs(signal))
    if peak_amplitude == 0:
        peak_amplitude = 1e-12

    normalized_signal = signal / peak_amplitude

    level_indices = np.round((normalized_signal + 1) / 2 * (num_levels - 1)).astype(int)
    level_indices = np.clip(level_indices, 0, num_levels - 1)

    quantized_normalized = level_indices / (num_levels - 1) * 2 - 1
    quantized_signal = quantized_normalized * peak_amplitude

    quantization_error = signal - quantized_signal

    signal_power = np.mean(signal.astype(np.float64) ** 2)
    error_power = np.mean(quantization_error.astype(np.float64) ** 2)

    if error_power > 0:
        snr_db = 10 * np.log10(signal_power / error_power)
    else:
        snr_db = float("inf")

    print(f"Cuantización a {num_bits} bits ({num_levels} niveles): SNR = {snr_db:.2f} dB")

    return quantized_signal, level_indices, quantization_error, snr_db


def encode_signal(quantization_levels, num_bits, sample_rate=None):
    """
    Codifica los niveles de cuantización como código PCM binario.

    Parameters
    ----------
    quantization_levels : numpy.ndarray
        Índices enteros de cuantización (ver quantize_signal).
    num_bits : int
        Número de bits usados por muestra.
    sample_rate : int, optional
        Frecuencia de muestreo de la señal, en Hz. Si se proporciona, se
        calcula e imprime el bitrate resultante (sample_rate * num_bits).

    Returns
    -------
    codes : list of str
        Código binario (de num_bits bits) de cada muestra.
    bitstream : str
        Todos los códigos concatenados en un único string de bits.
    """
    codes = [format(int(level), f"0{num_bits}b") for level in quantization_levels]
    bitstream = "".join(codes)

    total_bits = len(bitstream)
    print(f"Tamaño total codificado: {total_bits} bits ({total_bits / 8:.0f} bytes)")

    if sample_rate is not None:
        bitrate = sample_rate * num_bits
        print(f"Bitrate resultante: {bitrate} bits/s ({bitrate / 1000:.2f} kbps)")

    return codes, bitstream


def plot_sampling_quantization(
    original_signal,
    quantized_signal,
    sample_rate,
    output_path=DEFAULT_SAMPLING_PLOT_PATH,
    zoom_seconds=0.02,
):
    """
    Grafica un acercamiento (zoom) de la señal original superpuesta con la
    señal cuantizada, para visualizar el efecto de la cuantización muestra
    a muestra.

    Parameters
    ----------
    original_signal : numpy.ndarray
        Señal antes de cuantizar (misma frecuencia de muestreo que quantized_signal).
    quantized_signal : numpy.ndarray
        Señal ya cuantizada.
    sample_rate : int
        Frecuencia de muestreo de ambas señales, en Hz.
    output_path : str, optional
        Ruta donde se guardará la gráfica generada.
    zoom_seconds : float, optional
        Duración, en segundos, de la ventana de zoom a graficar.
    """
    num_zoom_samples = max(int(zoom_seconds * sample_rate), 1)
    num_zoom_samples = min(num_zoom_samples, len(original_signal))

    time_axis = np.arange(num_zoom_samples) / sample_rate

    plt.figure(figsize=(10, 4))
    plt.plot(
        time_axis,
        original_signal[:num_zoom_samples],
        label="Señal original",
        linewidth=1.2,
    )
    plt.step(
        time_axis,
        quantized_signal[:num_zoom_samples],
        label="Señal cuantizada",
        where="mid",
        linewidth=1.2,
    )
    plt.title(f"Muestreo y cuantización (zoom de {zoom_seconds * 1000:.0f} ms)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()

    print(f"Gráfica guardada en: {output_path}")


def plot_quantization_comparison(
    signal,
    sample_rate,
    bit_depths=[8, 4, 2],
    output_path=DEFAULT_BIT_DEPTH_PLOT_PATH,
):
    """
    Compara visualmente el efecto de cuantizar la misma señal con distintas
    profundidades de bits, en subplots apilados.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio a cuantizar con cada profundidad de bits.
    sample_rate : int
        Frecuencia de muestreo de la señal, en Hz.
    bit_depths : list of int, optional
        Profundidades de bits a comparar.
    output_path : str, optional
        Ruta donde se guardará la gráfica generada.
    """
    time_axis = np.arange(len(signal)) / sample_rate

    fig, axes = plt.subplots(
        len(bit_depths), 1, figsize=(10, 3 * len(bit_depths)), sharex=True
    )
    if len(bit_depths) == 1:
        axes = [axes]

    for ax, num_bits in zip(axes, bit_depths):
        quantized_signal, _, _, snr_db = quantize_signal(signal, num_bits)
        ax.plot(time_axis, quantized_signal, linewidth=0.7)
        ax.set_title(f"{num_bits} bits ({2 ** num_bits} niveles) - SNR = {snr_db:.2f} dB")
        ax.set_ylabel("Amplitud")

    axes[-1].set_xlabel("Tiempo (s)")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path)
    plt.close(fig)

    print(f"Gráfica guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remuestrea, cuantiza y codifica un archivo de audio de voz."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(DEFAULT_AUDIO_PATH),
        help=f"Ruta al audio limpio a procesar (default: {DEFAULT_AUDIO_PATH})",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=DEFAULT_TARGET_SR,
        help=f"Frecuencia de muestreo objetivo, en Hz (default: {DEFAULT_TARGET_SR})",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=DEFAULT_NUM_BITS,
        help=f"Número de bits de cuantización (default: {DEFAULT_NUM_BITS})",
    )
    args = parser.parse_args()

    resolved_path = _resolve_input_path(args.audio_path)

    signal, orig_sr = load_audio(resolved_path)

    resampled_signal = resample_signal(signal, orig_sr, args.target_sr)

    quantized_signal, level_indices, _, snr_db = quantize_signal(
        resampled_signal, args.num_bits
    )

    _, _ = encode_signal(level_indices, args.num_bits, sample_rate=args.target_sr)

    output_audio_path = DEFAULT_QUANTIZED_AUDIO_PATH
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_audio_path, quantized_signal, args.target_sr)
    print(f"Señal cuantizada guardada en: {output_audio_path}")

    plot_sampling_quantization(resampled_signal, quantized_signal, args.target_sr)
    plot_quantization_comparison(resampled_signal, args.target_sr)

    bitrate = args.target_sr * args.num_bits

    print("\nResumen final")
    print("-------------")
    print(f"Sample rate original: {orig_sr} Hz")
    print(f"Sample rate final:    {args.target_sr} Hz")
    print(f"Bits por muestra:     {args.num_bits}")
    print(f"SNR de cuantización:  {snr_db:.2f} dB")
    print(f"Bitrate resultante:   {bitrate} bits/s ({bitrate / 1000:.2f} kbps)")


if __name__ == "__main__":
    main()
