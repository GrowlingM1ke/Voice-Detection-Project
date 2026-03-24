"""
VAD-based chunk extraction and manifest generation.

Processes VCTK, VoxCeleb1, and VOiCES datasets:
  - Resamples all audio to 16 kHz mono
  - Runs Silero VAD to identify speech regions
  - Extracts 3-second chunks (configurable stride / overlap)
  - Drops chunks with insufficient speech content
  - Assigns speakers entirely to train / val / test (open-set protocol)
  - Saves chunk wavs and writes a manifest CSV
"""

import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from voice_detection.data_loading import load_vctk, load_voxceleb1, load_voices

TARGET_SR = 16_000
MIN_SPEECH_RATIO = 0.8  # ≥80% of a chunk must be speech to keep it


# ---------------------------------------------------------------------------
# VAD helpers
# ---------------------------------------------------------------------------

def load_vad_model():
    """Load and return the Silero VAD model."""
    from silero_vad import load_silero_vad
    return load_silero_vad()


def _speech_mask(wav: torch.Tensor, model) -> torch.Tensor:
    """Boolean tensor (per sample) — True where VAD detects speech."""
    from silero_vad import get_speech_timestamps
    timestamps = get_speech_timestamps(wav, model, sampling_rate=TARGET_SR)
    mask = torch.zeros(wav.shape[-1], dtype=torch.bool)
    for ts in timestamps:
        mask[ts["start"]: ts["end"]] = True
    return mask


# ---------------------------------------------------------------------------
# Speaker-level train / val / test split
# ---------------------------------------------------------------------------

def assign_speaker_splits(
    samples: list[tuple[Path, str]],
    dataset_name: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[tuple[Path, str, str, str]]:
    """
    Assign every speaker entirely to one split (open-set protocol).

    Speaker IDs are namespaced with the dataset name to avoid cross-dataset
    collisions (e.g. "vctk_p225", "vox1_id10001", "voices_sp0032").

    Args:
        samples:      Output of a load_* function — list of (wav_path, speaker_id).
        dataset_name: Short name used to namespace speaker IDs ("vctk", "vox1", "voices").
        val_ratio:    Fraction of speakers assigned to validation.
        test_ratio:   Fraction of speakers assigned to test.
        seed:         Random seed for reproducibility.

    Returns:
        List of (wav_path, namespaced_speaker_id, dataset_name, split).
    """
    rng = random.Random(seed)
    speakers = sorted({spk for _, spk in samples})
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))

    test_set = set(speakers[:n_test])
    val_set = set(speakers[n_test: n_test + n_val])

    result = []
    for wav_path, spk in samples:
        ns_spk = f"{dataset_name}_{spk}"
        if spk in test_set:
            split = "test"
        elif spk in val_set:
            split = "val"
        else:
            split = "train"
        result.append((wav_path, ns_spk, dataset_name, split))
    return result


# ---------------------------------------------------------------------------
# Per-file chunk extraction
# ---------------------------------------------------------------------------

def extract_chunks_from_file(
    wav_path: Path,
    speaker_id: str,
    dataset: str,
    split: str,
    out_dir: Path,
    vad_model,
    stride_sec: float = 1.5,
    chunk_sec: float = 3.0,
) -> list[dict]:
    """
    Load, resample, VAD-filter, and slice one audio file into fixed-length chunks.

    Args:
        wav_path:   Path to the source wav file.
        speaker_id: Namespaced speaker ID (e.g. "vctk_p225").
        dataset:    Dataset name ("vctk", "vox1", "voices").
        split:      "train", "val", or "test".
        out_dir:    Root directory under which chunks are saved.
        vad_model:  Loaded Silero VAD model.
        stride_sec: Step between chunk start times in seconds.
        chunk_sec:  Duration of each chunk in seconds (default 3.0).

    Returns:
        List of manifest row dicts with keys:
        chunk_path, speaker_id, dataset, split.
    """
    chunk_samples = int(TARGET_SR * chunk_sec)
    stride_samples = int(TARGET_SR * stride_sec)

    try:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        # soundfile returns (T, C) — convert to (C, T) torch tensor
        waveform = torch.from_numpy(data.T)
    except Exception as e:
        print(f"[WARN] Failed to load {wav_path}: {e}")
        return []

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    wav = waveform.squeeze(0)  # (T,)

    if wav.shape[0] < chunk_samples:
        return []

    mask = _speech_mask(wav, vad_model)

    chunk_dir = out_dir / split / dataset / speaker_id
    chunk_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    chunk_idx = 0
    start = 0

    while start + chunk_samples <= wav.shape[0]:
        end = start + chunk_samples
        if mask[start:end].float().mean().item() >= MIN_SPEECH_RATIO:
            chunk = wav[start:end].numpy()  # (T,)
            out_path = chunk_dir / f"{wav_path.stem}_{chunk_idx:04d}.wav"
            sf.write(str(out_path), chunk, TARGET_SR)
            rows.append({
                "chunk_path": str(out_path),
                "speaker_id": speaker_id,
                "dataset": dataset,
                "split": split,
            })
            chunk_idx += 1
        start += stride_samples

    return rows


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    out_dir: str | Path,
    manifest_path: str | Path,
    vctk_root: str | Path | None = None,
    voxceleb1_root: str | Path | None = None,
    voices_root: str | Path | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stride_sec: float = 1.5,
    chunk_sec: float = 3.0,
    seed: int = 42,
) -> list[dict]:
    """
    Full preprocessing pipeline: load datasets → assign splits → extract chunks
    → save wavs → write manifest CSV.

    Args:
        out_dir:         Root directory for saved chunk wavs.
        manifest_path:   Output path for the manifest CSV.
        vctk_root:       Path to VCTK-Corpus/ (contains wav48/). None to skip.
        voxceleb1_root:  Path to vox1_dev_wav/ (contains wav/). None to skip.
        voices_root:     Path to outer VOiCES_devkit/. None to skip.
        val_ratio:       Fraction of speakers held out for validation.
        test_ratio:      Fraction of speakers held out for testing.
        stride_sec:      Stride between chunk windows in seconds.
        chunk_sec:       Duration of each chunk in seconds (default 3.0).
        seed:            Random seed for reproducible speaker splits.

    Returns:
        List of all manifest row dicts written to the CSV.
    """
    out_dir = Path(out_dir)
    manifest_path = Path(manifest_path)

    if not any([vctk_root, voxceleb1_root, voices_root]):
        raise ValueError("At least one dataset root must be provided.")

    # ---- collect samples from each dataset ----
    all_samples: list[tuple[Path, str, str, str]] = []

    if vctk_root:
        raw = load_vctk(vctk_root)
        all_samples.extend(assign_speaker_splits(raw, "vctk", val_ratio, test_ratio, seed))
        print(f"VCTK:      {len(raw):>7,} files, {len({s for _, s in raw}):>4} speakers")

    if voxceleb1_root:
        raw = load_voxceleb1(voxceleb1_root)
        all_samples.extend(assign_speaker_splits(raw, "vox1", val_ratio, test_ratio, seed))
        print(f"VoxCeleb1: {len(raw):>7,} files, {len({s for _, s in raw}):>4} speakers")

    if voices_root:
        raw = load_voices(voices_root)
        all_samples.extend(assign_speaker_splits(raw, "voices", val_ratio, test_ratio, seed))
        print(f"VOiCES:    {len(raw):>7,} files, {len({s for _, s in raw}):>4} speakers")

    print(f"\nTotal files to process: {len(all_samples):,}")
    print(f"Chunk size: {chunk_sec} s | Stride: {stride_sec} s | Min speech ratio: {MIN_SPEECH_RATIO}")

    # ---- load VAD model once ----
    print("\nLoading Silero VAD model...")
    vad_model = load_vad_model()

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- extract chunks ----
    all_rows: list[dict] = []
    for wav_path, ns_spk, dataset, split in tqdm(all_samples, desc="Extracting chunks"):
        rows = extract_chunks_from_file(
            wav_path, ns_spk, dataset, split,
            out_dir, vad_model, stride_sec, chunk_sec,
        )
        all_rows.extend(rows)

    # ---- write manifest ----
    fieldnames = ["chunk_path", "speaker_id", "dataset", "split"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ---- summary ----
    split_counts = Counter(r["split"] for r in all_rows)
    print(f"\nDone. Total chunks: {len(all_rows):,}")
    print(f"\n{'Split':<8} {'Chunks':>10} {'Speakers':>10}")
    print("-" * 30)
    for split in ("train", "val", "test"):
        count = split_counts.get(split, 0)
        speakers = len({r["speaker_id"] for r in all_rows if r["split"] == split})
        print(f"{split:<8} {count:>10,} {speakers:>10,}")
    print(f"\nManifest saved to: {manifest_path}")

    return all_rows
