"""PyTorch Dataset for speaker speech chunks."""

from __future__ import annotations

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
