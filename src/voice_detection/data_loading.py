"""
Data loading utilities for VCTK, VoxCeleb1, and VOiCES datasets.

Each function returns a list of (wav_path, speaker_id) tuples.
No audio is loaded here — this is purely path/metadata discovery.
"""

from pathlib import Path


def load_vctk(root: str | Path) -> list[tuple[Path, str]]:
    """
    Load VCTK Corpus wav paths and speaker IDs.

    Expected structure:
        <root>/wav48/<speaker_id>/<speaker_id>_<utt_id>.wav

    Args:
        root: Path to the VCTK-Corpus directory (the one containing wav48/).

    Returns:
        List of (wav_path, speaker_id) tuples.
    """
    root = Path(root)
    wav_root = root / "wav48"

    if not wav_root.exists():
        raise FileNotFoundError(f"wav48 directory not found at {wav_root}")

    samples: list[tuple[Path, str]] = []
    for speaker_dir in sorted(wav_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name  # e.g. "p225"
        for wav_file in sorted(speaker_dir.glob("*.wav")):
            samples.append((wav_file, speaker_id))

    return samples


def load_voxceleb1(root: str | Path) -> list[tuple[Path, str]]:
    """
    Load VoxCeleb1 wav paths and speaker IDs.

    Expected structure:
        <root>/wav/<speaker_id>/<video_id>/<utt_id>.wav

    Args:
        root: Path to the vox1_dev_wav directory (the one containing wav/).

    Returns:
        List of (wav_path, speaker_id) tuples.
    """
    root = Path(root)
    wav_root = root / "wav"

    if not wav_root.exists():
        raise FileNotFoundError(f"wav directory not found at {wav_root}")

    samples: list[tuple[Path, str]] = []
    for speaker_dir in sorted(wav_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name  # e.g. "id10001"
        for wav_file in sorted(speaker_dir.rglob("*.wav")):
            samples.append((wav_file, speaker_id))

    return samples


def load_voices(root: str | Path) -> list[tuple[Path, str]]:
    """
    Load VOiCES dataset wav paths and speaker IDs.

    Expected structure (source recordings only):
        <root>/VOiCES_devkit/source-16k/{train|test}/<speaker_id>/*.wav

    Speaker ID is taken directly from the parent directory name (e.g. "sp0032").
    Only source-16k is used; distant-16k (far-field) recordings are excluded.

    Args:
        root: Path to the outer VOiCES_devkit directory (the one extracted from
              the tarball — contains another VOiCES_devkit/ inside it).

    Returns:
        List of (wav_path, speaker_id) tuples.
    """
    root = Path(root)
    source_root = root / "VOiCES_devkit" / "source-16k"

    if not source_root.exists():
        raise FileNotFoundError(f"source-16k directory not found at {source_root}")

    samples: list[tuple[Path, str]] = []
    for split_dir in sorted(source_root.iterdir()):  # train / test
        if not split_dir.is_dir():
            continue
        for speaker_dir in sorted(split_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name  # e.g. "sp0032"
            for wav_file in sorted(speaker_dir.glob("*.wav")):
                samples.append((wav_file, speaker_id))

    return samples
