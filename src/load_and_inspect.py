"""
load_and_inspect.py

Primer módulo del pipeline de procesamiento de audio.

Carga un archivo de voz ya grabado (wav, mp3, m4a, etc.), imprime un resumen
de sus características básicas y grafica la forma de onda en el dominio del
tiempo.
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_AUDIO_PATH = PROJECT_ROOT / "audio" / "raw_recording.wav"
DEFAULT_WAVEFORM_PLOT_PATH = PROJECT_ROOT / "figures" / "01_raw_waveform.png"


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
        f"  - Relativo a la raíz del proyecto: {project_root_path}\n"
        "Coloca tu grabación de voz en la carpeta 'audio/' o indica la "
        "ruta correcta."
    )


def load_audio(path):
    """
    Carga un archivo de audio desde disco usando librosa.

    Se usa sr=None para conservar el sample rate original del archivo
    (librosa no lo re-muestrea).

    Parameters
    ----------
    path : str or Path
        Ruta al archivo de audio (wav, mp3, m4a u otro formato soportado
        por librosa).

    Returns
    -------
    signal : numpy.ndarray
        Señal de audio como arreglo de numpy (mono).
    sample_rate : int
        Frecuencia de muestreo original del archivo, en Hz.

    Raises
    ------
    FileNotFoundError
        Si la ruta indicada no existe.
    RuntimeError
        Si el archivo existe pero no pudo ser leído/decodificado.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de audio: '{path}'. "
            "Coloca tu grabación de voz en la carpeta 'audio/' o indica "
            "la ruta correcta como argumento."
        )

    try:
        signal, sample_rate = librosa.load(path, sr=None)
    except Exception as error:
        raise RuntimeError(
            f"No se pudo leer el archivo de audio '{path}'. "
            f"Verifica que el formato sea válido (wav, mp3, m4a, ...). "
            f"Detalle del error: {error}"
        ) from error

    return signal, sample_rate


def inspect_audio(signal, sample_rate):
    """
    Imprime un resumen de las características básicas de la señal de audio.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio.
    sample_rate : int
        Frecuencia de muestreo de la señal, en Hz.
    """
    num_samples = len(signal)
    duration_seconds = num_samples / sample_rate
    max_amplitude = np.max(signal)
    min_amplitude = np.min(signal)
    rms_value = np.sqrt(np.mean(signal.astype(np.float64) ** 2))

    print("Resumen del audio")
    print("-----------------")
    print(f"Duración:           {duration_seconds:.3f} s")
    print(f"Número de muestras: {num_samples}")
    print(f"Sample rate:        {sample_rate} Hz")
    print(f"Amplitud máxima:    {max_amplitude:.4f}")
    print(f"Amplitud mínima:    {min_amplitude:.4f}")
    print(f"Valor RMS:          {rms_value:.4f}")


def plot_waveform(signal, sample_rate, output_path=DEFAULT_WAVEFORM_PLOT_PATH):
    """
    Grafica la forma de onda de la señal en el dominio del tiempo y la
    guarda como imagen PNG.

    Parameters
    ----------
    signal : numpy.ndarray
        Señal de audio.
    sample_rate : int
        Frecuencia de muestreo de la señal, en Hz.
    output_path : str or Path, optional
        Ruta donde se guardará la gráfica generada. Por defecto, dentro de
        la carpeta 'figures/' de la raíz del proyecto.
    """
    time_axis = np.arange(len(signal)) / sample_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal, linewidth=0.5)
    plt.title("Forma de onda de la señal de audio")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()

    print(f"Gráfica guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Carga e inspecciona un archivo de audio de voz."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(DEFAULT_AUDIO_PATH),
        help=f"Ruta al archivo de audio a procesar (default: {DEFAULT_AUDIO_PATH})",
    )
    args = parser.parse_args()

    resolved_path = _resolve_input_path(args.audio_path)

    signal, sample_rate = load_audio(resolved_path)
    inspect_audio(signal, sample_rate)
    plot_waveform(signal, sample_rate)


if __name__ == "__main__":
    main()
