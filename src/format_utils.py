"""
format_utils.py

Utilidades de conversión de formato de audio.

Permite convertir archivos de audio en formatos comunes (OGG, MP3, M4A, ...)
a WAV, que es el formato que usan los demás módulos del pipeline.
"""

import argparse
from pathlib import Path

import soundfile as sf

import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_AUDIO_DIR = PROJECT_ROOT / "audio"


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


def convert_to_wav(input_path, output_path=None):
    """
    Convierte un archivo de audio a formato WAV.

    Carga el archivo de entrada con librosa (que soporta OGG, MP3, M4A, WAV
    y otros formatos vía libsndfile/audioread) usando sr=None para conservar
    el sample rate original, y lo guarda como WAV con soundfile.

    Parameters
    ----------
    input_path : str or Path
        Ruta al archivo de audio de entrada.
    output_path : str or Path, optional
        Ruta del WAV de salida. Si no se especifica, se usa el mismo nombre
        base del archivo de entrada con extensión .wav, dentro de la
        carpeta 'audio/' de la raíz del proyecto.

    Returns
    -------
    Path
        Ruta del archivo WAV generado.

    Raises
    ------
    FileNotFoundError
        Si el archivo de entrada no existe.
    RuntimeError
        Si librosa no pudo decodificar el archivo (códec no soportado),
        con instrucciones para convertirlo manualmente usando ffmpeg.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de audio: '{input_path}'.")

    if output_path is None:
        output_path = DEFAULT_AUDIO_DIR / f"{input_path.stem}.wav"
    output_path = Path(output_path)

    try:
        signal, sample_rate = librosa.load(input_path, sr=None)
    except Exception as error:
        raise RuntimeError(
            f"No se pudo decodificar '{input_path}' (posible códec no soportado). "
            "Instala ffmpeg (https://ffmpeg.org/download.html) y convierte el "
            "archivo manualmente, por ejemplo: "
            f"ffmpeg -i \"{input_path}\" \"{output_path}\". "
            f"Detalle del error: {error}"
        ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(output_path, signal, sample_rate)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convierte un archivo de audio a formato WAV."
    )
    parser.add_argument("audio_path", help="Ruta al archivo de audio a convertir")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Ruta del WAV de salida (default: audio/<mismo_nombre>.wav en la raíz del proyecto)",
    )
    args = parser.parse_args()

    resolved_path = _resolve_input_path(args.audio_path)

    output_path = convert_to_wav(resolved_path, args.output)
    print(f"Archivo convertido a WAV: {output_path}")


if __name__ == "__main__":
    main()
