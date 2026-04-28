"""LeJEPA training script for speaker verification."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import brentq
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from voice_detection.dataset import SpeakerChunkDataset, SpeakerMultiViewDataset
from voice_detection.model import SIGReg, ViTEncoder


# ---------------------------------------------------------------------------
# Embedding visualisation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def visualize_embeddings(
    net: ViTEncoder,
    val_ds: SpeakerChunkDataset,
    device: torch.device,
    save_path: str | Path,
    num_speakers: int = 10,
    chunks_per_speaker: int = 50,
) -> None:
    """UMAP scatter + cosine similarity heatmap for a random speaker subset.

    Overwrites *save_path* on every call so only the latest visualisation
    (corresponding to the current best.pt) is kept.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    net.eval()

    # Pick random speakers that have enough chunks
    all_speakers = sorted(val_ds.speaker_to_idx.keys())
    rng = random.Random(0)
    rng.shuffle(all_speakers)
    chosen = all_speakers[:num_speakers]
    idx_map = {spk: val_ds.speaker_to_idx[spk] for spk in chosen}

    embs_list: list[np.ndarray] = []
    labels_list: list[str] = []

    for spk in chosen:
        rows = val_ds.df[val_ds.df["speaker_id"] == spk].head(chunks_per_speaker)
        for _, row in rows.iterrows():
            import soundfile as sf
            data, _ = sf.read(row["chunk_path"], dtype="float32")
            waveform = torch.from_numpy(data).unsqueeze(0)
            rms = waveform.square().mean().sqrt()
            waveform = waveform / (rms + 1e-9)
            mel = val_ds._mel(waveform)
            log_mel = torch.log(mel + 1e-9)
            if val_ds._normalize:
                log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
            spec = log_mel.unsqueeze(0)[..., :300].to(device)
            emb = net.embed(spec)
            emb = F.normalize(emb.float(), dim=1).cpu().numpy()
            embs_list.append(emb[0])
            labels_list.append(spk)

    embs = np.array(embs_list)          # (N, embed_dim)
    labels_arr = np.array(labels_list)

    # UMAP projection
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embs)  # (N, 2)

    # Cosine similarity between speaker centroids
    centroids = np.stack(
        [embs[labels_arr == spk].mean(0) for spk in chosen]
    )
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / (norms + 1e-9)
    sim_matrix = centroids_norm @ centroids_norm.T  # (num_speakers, num_speakers)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- UMAP scatter ---
    cmap = plt.cm.get_cmap("tab10", num_speakers)
    for i, spk in enumerate(chosen):
        mask = labels_arr == spk
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=12, color=cmap(i), label=spk, alpha=0.7)
    axes[0].set_title("UMAP of speaker embeddings (val)")
    axes[0].legend(markerscale=2, fontsize=7, loc="best")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    # --- Cosine similarity heatmap ---
    im = axes[1].imshow(sim_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    short_labels = [spk.split("_", 1)[-1] for spk in chosen]
    axes[1].set_xticks(range(num_speakers))
    axes[1].set_yticks(range(num_speakers))
    axes[1].set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(short_labels, fontsize=8)
    axes[1].set_title("Cosine similarity — speaker centroids")
    fig.colorbar(im, ax=axes[1])

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    manifest_path: str = "data/manifest.csv"
    num_views: int = 2
    min_chunks: int = 10
    batch_size: int = 128
    num_workers: int = 4

    # Model
    embed_dim: int = 512
    proj_dim: int = 128

    # Training
    epochs: int = 10
    lr: float = 5e-4
    weight_decay: float = 5e-2
    lamb: float = 0.5           # balance: SIGReg * lamb + invariance * (1 - lamb)
    warmup_epochs: int = 1
    seed: int = 42

    # System
    device: str = "cuda"

    # Logging / checkpointing
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5         # save checkpoint every N epochs
    eval_every: int = 1         # run EER evaluation every N epochs
    num_eval_pairs: int = 10_000
    max_eval_chunks: int = 5_000


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eer(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Compute Equal Error Rate from positive and negative cosine similarity scores."""
    pos = np.sort(pos_scores)
    neg = np.sort(neg_scores)
    n_pos = len(pos)
    n_neg = len(neg)

    def far_minus_frr(threshold: float) -> float:
        far = 1.0 - np.searchsorted(neg, threshold, side="left") / n_neg
        frr = np.searchsorted(pos, threshold, side="left") / n_pos
        return far - frr

    lo = min(pos[0], neg[0]) - 1e-6
    hi = max(pos[-1], neg[-1]) + 1e-6

    try:
        threshold = brentq(far_minus_frr, lo, hi)
        far = (neg >= threshold).mean()
        frr = (pos < threshold).mean()
        return float((far + frr) / 2)
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_eer(
    net: ViTEncoder,
    val_loader: DataLoader,
    device: torch.device,
    num_pairs: int = 10_000,
    max_chunks: int = 5_000,
) -> float:
    """Extract backbone embeddings from val set and compute EER."""
    net.eval()

    all_embs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    n_chunks = 0

    for spec, label in val_loader:
        spec = spec[..., :300].to(device)  # trim to 300 frames
        with autocast(device.type, dtype=torch.bfloat16):
            emb = net.embed(spec)
        all_embs.append(F.normalize(emb.float(), dim=1).cpu())
        all_labels.append(label)
        n_chunks += len(label)
        if n_chunks >= max_chunks:
            break

    embs = torch.cat(all_embs).numpy()
    labels = torch.cat(all_labels).numpy()

    # Build speaker -> indices
    unique_labels = np.unique(labels)
    label_to_idx = {int(s): np.where(labels == s)[0] for s in unique_labels}

    pos_scores = []
    neg_scores = []

    for _ in range(num_pairs):
        spk = int(np.random.choice(unique_labels))
        indices = label_to_idx[spk]
        if len(indices) < 2:
            continue

        i, j = np.random.choice(indices, size=2, replace=False)
        pos_scores.append(float((embs[i] * embs[j]).sum()))

        # Negative pair
        other = np.random.choice(unique_labels[unique_labels != spk])
        k = np.random.choice(label_to_idx[int(other)])
        neg_scores.append(float((embs[i] * embs[k]).sum()))

    if len(pos_scores) < 100:
        return 0.5

    return compute_eer(np.array(pos_scores), np.array(neg_scores))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    # ---- Data ----
    train_ds = SpeakerMultiViewDataset(
        cfg.manifest_path,
        "train",
        num_views=cfg.num_views,
        min_chunks=cfg.min_chunks,
    )
    val_ds = SpeakerChunkDataset(cfg.manifest_path, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    print(f"Train: {len(train_ds):,} chunks, {train_ds.num_speakers} speakers")
    print(f"Val:   {len(val_ds):,} chunks, {val_ds.num_speakers} speakers")

    # ---- Model ----
    net = ViTEncoder(embed_dim=cfg.embed_dim, proj_dim=cfg.proj_dim).to(device)
    probe = nn.Sequential(
        nn.LayerNorm(cfg.embed_dim),
        nn.Linear(cfg.embed_dim, train_ds.num_speakers),
    ).to(device)
    sigreg = SIGReg().to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Encoder parameters: {total_params:,}")

    # ---- Optimizer & Scheduler ----
    opt = torch.optim.AdamW(
        [
            {"params": net.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7},
        ]
    )
    warmup_steps = len(train_loader) * cfg.warmup_epochs
    total_steps = len(train_loader) * cfg.epochs
    scheduler = SequentialLR(
        opt,
        [
            LinearLR(opt, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-5),
        ],
        milestones=[warmup_steps],
    )
    scaler = GradScaler()

    # ---- Training ----
    global_step = 0
    best_eer = 1.0

    for epoch in range(cfg.epochs):
        net.train()
        probe.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for views, labels in pbar:
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device.type, dtype=torch.bfloat16):
                emb, proj = net(views)

                # LeJEPA losses
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)

                # Speaker classification (trains backbone directly)
                labels_rep = labels.repeat_interleave(cfg.num_views)
                probe_loss = F.cross_entropy(probe(emb), labels_rep)

                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lejepa", lejepa_loss.item(), global_step)
            writer.add_scalar("train/invariance", inv_loss.item(), global_step)
            writer.add_scalar("train/sigreg", sigreg_loss.item(), global_step)
            writer.add_scalar("train/probe_loss", probe_loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                inv=f"{inv_loss.item():.4f}",
                sig=f"{sigreg_loss.item():.4f}",
            )
            global_step += 1

        # ---- Evaluation ----
        if (epoch + 1) % cfg.eval_every == 0:
            eer = evaluate_eer(
                net, val_loader, device, cfg.num_eval_pairs, cfg.max_eval_chunks
            )
            writer.add_scalar("val/eer", eer, epoch + 1)
            print(f"  EER: {eer:.4f}")

            if eer < best_eer:
                best_eer = eer
                torch.save(net.state_dict(), Path(cfg.checkpoint_dir) / "best.pt")
                print(f"  New best EER — saved best.pt")
                viz_path = Path(cfg.log_dir) / "embedding_viz.png"
                visualize_embeddings(net, val_ds, device, viz_path)
                print(f"  Embedding visualisation saved to {viz_path}")

        # ---- Periodic checkpoint ----
        if (epoch + 1) % cfg.save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "net": net.state_dict(),
                "probe": probe.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
            }
            path = Path(cfg.checkpoint_dir) / f"epoch_{epoch + 1}.pt"
            torch.save(ckpt, path)
            print(f"  Saved checkpoint: {path}")

    writer.close()
    print(f"Training complete. Best EER: {best_eer:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train LeJEPA speaker encoder")
    for name in cfg.__dataclass_fields__:
        val = getattr(cfg, name)
        parser.add_argument(f"--{name}", type=type(val), default=val)
    args = parser.parse_args()
    cfg = Config(**vars(args))

    train(cfg)
