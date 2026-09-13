"""
main.py

Orquestador end-to-end del pipeline de procesamiento de audio.

Encadena, en orden, todos los módulos ya existentes del proyecto:
carga/inspección, filtrado de silencios, muestreo/cuantización/codificación
y análisis en frecuencia. No reimplementa ninguna lógica de procesamiento de
señal: únicamente importa y llama a las funciones ya probadas en cada
módulo individual, guarda los archivos de audio intermedios y finales, y
consolida los resultados en un único resumen impreso al final.
"""

import argparse
from pathlib import Path

import soundfile as sf

import format_utils
from load_and_inspect import load_audio, inspect_audio, plot_waveform
from preprocess import remove_silence, plot_silence_removal
from sampling_quant_code import (
    resample_signal,
    quantize_signal,
    encode_signal,
    plot_sampling_quantization,
    plot_quantization_comparison,
)
from fourier_analysis import (
    compute_fft,
    plot_spectrum,
    compute_spectrogram_dominant_freqs,
    plot_frequency_histogram,
    identify_dominant_frequency,
    identify_frequency_range,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_AUDIO_DIR = PROJECT_ROOT / "audio"
DEFAULT_CLEANED_AUDIO_PATH = PROJECT_ROOT / "audio" / "cleaned_recording.wav"
DEFAULT_QUANTIZED_AUDIO_PATH = PROJECT_ROOT / "audio" / "quantized_recording.wav"

DEFAULT_TARGET_SR = 8000
DEFAULT_NUM_BITS = 8
DEFAULT_THRESHOLD_DB = -20

GENERATED_AUDIO_FILENAMES = {"cleaned_recording.wav", "quantized_recording.wav"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


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


def _find_default_input_audio():
    """
    Busca, dentro de la carpeta 'audio/' del proyecto, el primer archivo de
    audio que no haya sido generado por el propio pipeline (es decir,
    distinto de cleaned_recording.wav y quantized_recording.wav), para
    usarlo como entrada por defecto cuando el usuario no indica una ruta.

    Returns
    -------
    Path
        Ruta del archivo de audio de entrada encontrado.

    Raises
    ------
    FileNotFoundError
        Si la carpeta 'audio/' no existe o no contiene ningún archivo de
        audio de entrada válido.
    """
    if not DEFAULT_AUDIO_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de audio: '{DEFAULT_AUDIO_DIR}'. "
            "Crea la carpeta y coloca ahí tu grabación de voz."
        )

    candidates = sorted(
        path
        for path in DEFAULT_AUDIO_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and path.name not in GENERATED_AUDIO_FILENAMES
    )

    if not candidates:
        raise FileNotFoundError(
            f"No se encontró ningún archivo de audio de entrada en "
            f"'{DEFAULT_AUDIO_DIR}'. Coloca tu grabación de voz ahí, o "
            "indica la ruta explícitamente como argumento."
        )

    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline completo de procesamiento de audio, "
        "de principio a fin."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=None,
        help="Ruta al archivo de audio crudo de entrada (default: el primer "
        "archivo de audio encontrado en audio/ que no sea generado por el "
        "propio pipeline)",
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
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=DEFAULT_THRESHOLD_DB,
        help=f"Umbral, en dB relativos al pico, para el rango de frecuencias "
        f"(default: {DEFAULT_THRESHOLD_DB})",
    )
    args = parser.parse_args()

    if args.audio_path is None:
        resolved_path = _find_default_input_audio()
    else:
        resolved_path = _resolve_input_path(args.audio_path)

    if resolved_path.suffix.lower() != ".wav":
        wav_path = format_utils.convert_to_wav(resolved_path)
        print(f"Archivo convertido a WAV: {wav_path}")
    else:
        wav_path = resolved_path

    signal, orig_sr = load_audio(wav_path)
    inspect_audio(signal, orig_sr)
    plot_waveform(signal, orig_sr)

    clean_signal, intervals, top_db = remove_silence(signal, orig_sr)

    DEFAULT_CLEANED_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(DEFAULT_CLEANED_AUDIO_PATH, clean_signal, orig_sr)
    print(f"Señal limpia guardada en: {DEFAULT_CLEANED_AUDIO_PATH}")

    plot_silence_removal(signal, orig_sr, intervals)

    original_duration = len(signal) / orig_sr
    clean_duration = len(clean_signal) / orig_sr
    percent_removed = (
        (1 - clean_duration / original_duration) * 100 if original_duration > 0 else 0.0
    )

    resampled_signal = resample_signal(clean_signal, orig_sr, args.target_sr)

    quantized_signal, level_indices, _, snr_db = quantize_signal(
        resampled_signal, args.num_bits
    )
    encode_signal(level_indices, args.num_bits, sample_rate=args.target_sr)

    DEFAULT_QUANTIZED_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(DEFAULT_QUANTIZED_AUDIO_PATH, quantized_signal, args.target_sr)
    print(f"Señal cuantizada guardada en: {DEFAULT_QUANTIZED_AUDIO_PATH}")

    plot_sampling_quantization(resampled_signal, quantized_signal, args.target_sr)
    plot_quantization_comparison(resampled_signal, args.target_sr)

    freqs, _, magnitude_db = compute_fft(quantized_signal, args.target_sr)
    plot_spectrum(freqs, magnitude_db, args.target_sr)

    dominant_freqs_per_frame = compute_spectrogram_dominant_freqs(
        quantized_signal, args.target_sr
    )
    plot_frequency_histogram(dominant_freqs_per_frame)

    peak_freq, mode_freq = identify_dominant_frequency(
        freqs, magnitude_db, dominant_freqs_per_frame
    )
    freq_min, freq_max = identify_frequency_range(
        freqs, magnitude_db, threshold_db=args.threshold_db
    )

    bitrate = args.target_sr * args.num_bits

    print("\n==================== RESUMEN CONSOLIDADO ====================")
    print(f"Archivo de entrada:               {resolved_path}")
    print(f"Duración original:                {original_duration:.3f} s")
    print(f"Audio removido como silencio:     {percent_removed:.1f}%")
    print(f"Sample rate original:             {orig_sr} Hz")
    print(f"Sample rate final:                {args.target_sr} Hz")
    print(f"Bits por muestra:                 {args.num_bits}")
    print(f"SNR de cuantización:              {snr_db:.2f} dB")
    print(f"Bitrate resultante:               {bitrate} bits/s ({bitrate / 1000:.2f} kbps)")
    print(f"Frecuencia dominante (pico FFT):  {peak_freq:.1f} Hz")
    print(f"Frecuencia dominante (moda):      {mode_freq:.1f} Hz")
    print(f"Rango de frecuencias detectado:   {freq_min:.1f} Hz - {freq_max:.1f} Hz")
    print("===============================================================")


if __name__ == "__main__":
    main()
