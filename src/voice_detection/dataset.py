"""PyTorch Dataset for speaker speech chunks."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class SpeakerChunkDataset(Dataset):
    """
    Loads 3-second speech chunks from a manifest CSV and returns
    log-mel spectrogram tensors paired with integer speaker labels.

    Args:
        manifest_path:  Path to manifest.csv produced by run_preprocessing().
        split:          One of "train", "val", or "test".
        n_mels:         Number of mel filter banks (default 80).
        n_fft:          FFT window size in samples (default 400 = 25 ms @ 16 kHz).
        hop_length:     Hop length in samples (default 160 = 10 ms @ 16 kHz).
        sample_rate:    Expected sample rate of the chunk WAV files.
        normalize:      If True, apply per-chunk mean/std normalisation to the
                        log-mel spectrogram (stabilises training).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        sample_rate: int = 16_000,
        normalize: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        manifest = pd.read_csv(manifest_path)
        self.df = manifest[manifest["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split!r} in {manifest_path}")

        # Stable speaker → integer label mapping (sorted for reproducibility)
        speakers = sorted(self.df["speaker_id"].unique())
        self.speaker_to_idx: dict[str, int] = {spk: i for i, spk in enumerate(speakers)}

        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._normalize = normalize

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        data, _ = sf.read(row["chunk_path"], dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)

        mel = self._mel(waveform)          # (1, n_mels, T_frames)
        log_mel = torch.log(mel + 1e-9)   # log compression

        if self._normalize:
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

        label = self.speaker_to_idx[row["speaker_id"]]
        return log_mel, label

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_speakers(self) -> int:
        """Number of unique speakers in this split."""
        return len(self.speaker_to_idx)

    @property
    def spectrogram_shape(self) -> tuple[int, int, int]:
        """
        Shape of one spectrogram tensor (C, n_mels, T_frames).
        Computed from the first item — requires at least one row.
        """
        spec, _ = self[0]
        return tuple(spec.shape)  # type: ignore[return-value]


class SpeakerMultiViewDataset(Dataset):
    """
    Multi-view dataset for self-supervised speaker representation learning.

    Each __getitem__ returns V log-mel spectrograms from the same speaker
    (different utterance chunks), enabling invariance-based learning.

    Args:
        manifest_path:  Path to manifest.csv produced by run_preprocessing().
        split:          One of "train", "val", or "test".
        num_views:      Number of different chunks to sample per speaker per item.
        min_chunks:     Minimum chunks a speaker must have to be included.
        target_frames:  Trim/pad spectrograms to this many time frames.
        n_mels:         Number of mel filter banks (default 80).
        n_fft:          FFT window size in samples (default 400 = 25 ms @ 16 kHz).
        hop_length:     Hop length in samples (default 160 = 10 ms @ 16 kHz).
        sample_rate:    Expected sample rate of the chunk WAV files.
        normalize:      If True, apply per-chunk mean/std normalisation.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        num_views: int = 2,
        min_chunks: int = 10,
        target_frames: int = 300,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        sample_rate: int = 16_000,
        normalize: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        manifest = pd.read_csv(manifest_path)
        df = manifest[manifest["split"] == split].reset_index(drop=True)

        # Filter speakers with too few chunks
        speaker_counts = df["speaker_id"].value_counts()
        valid_speakers = set(speaker_counts[speaker_counts >= min_chunks].index)
        self.df = df[df["speaker_id"].isin(valid_speakers)].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No rows for split={split!r} after filtering min_chunks={min_chunks}"
            )

        # Build speaker -> chunk indices mapping
        self._speaker_to_chunks: dict[str, list[int]] = {}
        for idx in range(len(self.df)):
            spk = self.df.at[idx, "speaker_id"]
            self._speaker_to_chunks.setdefault(spk, []).append(idx)

        # Stable speaker -> integer label mapping
        speakers = sorted(self._speaker_to_chunks.keys())
        self.speaker_to_idx: dict[str, int] = {spk: i for i, spk in enumerate(speakers)}

        self.num_views = num_views
        self._target_frames = target_frames
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._normalize = normalize

    def _load_spectrogram(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        data, _ = sf.read(row["chunk_path"], dtype="float32")
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)

        mel = self._mel(waveform)  # (1, n_mels, T_frames)
        log_mel = torch.log(mel + 1e-9)

        # Trim or pad to target_frames
        t = self._target_frames
        if log_mel.size(-1) > t:
            log_mel = log_mel[..., :t]
        elif log_mel.size(-1) < t:
            log_mel = torch.nn.functional.pad(log_mel, (0, t - log_mel.size(-1)))

        if self._normalize:
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

        return log_mel

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        speaker = row["speaker_id"]
        label = self.speaker_to_idx[speaker]

        spec = self._load_spectrogram(idx)

        if self.num_views == 1:
            return spec.unsqueeze(0), label

        # View 1: augmented version of this chunk
        view1 = self._augment(spec.clone())

        # View 2: augmented version of a *different* chunk from the same speaker
        pool = self._speaker_to_chunks[speaker]
        other_idx = idx
        if len(pool) > 1:
            others = [i for i in pool if i != idx]
            other_idx = random.choice(others)
        view2 = self._augment(self._load_spectrogram(other_idx))

        return torch.stack([view1, view2]), label

    def _augment(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply random time-frequency masking + gaussian noise to a spectrogram.

        Args:
            spec: (1, n_mels, T_frames) log-mel spectrogram.
        """
        _, n_mels, n_frames = spec.shape

        # Time masking: 1-3 masks of up to 30 frames each
        n_time_masks = random.randint(1, 3)
        for _ in range(n_time_masks):
            t = random.randint(1, min(30, n_frames // 5))
            t0 = random.randint(0, n_frames - t)
            spec[:, :, t0 : t0 + t] = 0.0

        # Frequency masking: 1-2 masks of up to 15 bins each
        n_freq_masks = random.randint(1, 2)
        for _ in range(n_freq_masks):
            f = random.randint(1, min(15, n_mels // 5))
            f0 = random.randint(0, n_mels - f)
            spec[:, f0 : f0 + f, :] = 0.0

        # Additive gaussian noise
        if random.random() < 0.5:
            spec = spec + torch.randn_like(spec) * random.uniform(0.01, 0.1)

        return spec

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)
