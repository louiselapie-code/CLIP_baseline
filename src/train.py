"""
Entraînement du CLIP ARN–protéine (baseline, §4–§6).

Entrée : les vues alignées produites par sanity_check.py --save (ou paired_data.save_paired) :
    paired_rna.npy (N, 64) + paired_protein.npy (N, P) + paired_cells.csv (cell_id, split)

- Entraîne uniquement la tête ARN + la tour protéine (NOVAE est figé en amont).
- AdamW + cosine decay (+ warmup optionnel), grad clipping, dropout (§5).
- Early stopping sur **Recall@5 val** (moyenne des deux directions), pas la val loss (§6.1).
- Monitoring (§6.2) : loss totale + 2 directions, Recall@1/5/10/50 + MedR sur val (2 sens),
  std par dim et cosinus intra-batch (détection de collapse). Logs CSV + courbes PNG.
- Checkpoints best (meilleur R@5 val) et last.

Le test n'est PAS touché ici (réservé à l'évaluation finale du §8).

Usage :
    python train.py --paired-dir MLP/paired --outdir MLP/runs/baseline_seed42
    # device auto (cuda > mps > cpu) ; surcharger avec --device cpu si besoin
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import CLIPConfig, CLIPModel, info_nce_symmetric
from protein_encoder import set_seed


# --------------------------------------------------------------------------- #
# Données / device
# --------------------------------------------------------------------------- #
def pick_device(arg: str) -> torch.device:
    if arg and arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_views(paired_dir: str):
    d = Path(paired_dir)
    rna = np.load(d / "paired_rna.npy").astype(np.float32)
    prot = np.load(d / "paired_protein.npy").astype(np.float32)
    split = pd.read_csv(d / "paired_cells.csv")["split"].to_numpy()
    return rna, prot, split


# --------------------------------------------------------------------------- #
# Évaluation : embeddings + retrieval + collapse
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_all(model, rna, prot, device, bs=8192):
    model.eval()
    zr, zp = [], []
    for s in range(0, len(rna), bs):
        r = torch.from_numpy(rna[s:s + bs]).to(device)
        p = torch.from_numpy(prot[s:s + bs]).to(device)
        a, b = model(r, p)
        zr.append(a.float().cpu().numpy())
        zp.append(b.float().cpu().numpy())
    return np.concatenate(zr), np.concatenate(zp)


def retrieval(Zq, Zg, ks=(1, 5, 10, 50), chunk=2048):
    """Zq, Zg déjà normalisés L2 (sortie du modèle). Rang du vrai appariement par cosinus."""
    N = len(Zq)
    ranks = np.empty(N, dtype=np.int64)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sims = Zq[s:e] @ Zg.T
        true = sims[np.arange(e - s), np.arange(s, e)][:, None]
        ranks[s:e] = (sims > true).sum(axis=1) + 1
    rec = {k: float(np.mean(ranks <= k)) for k in ks}
    return rec, float(np.median(ranks))


def collapse_stats(Zr, Zp, sample=4000, seed=0):
    """std par dimension (→0 = collapse) et cosinus intra-modalité moyen (→1 = collapse)."""
    idx = np.random.default_rng(seed).choice(len(Zr), size=min(sample, len(Zr)), replace=False)
    zr, zp = Zr[idx], Zp[idx]

    def mean_offdiag_cos(z):
        S = z @ z.T
        n = len(z)
        return float((S.sum() - np.trace(S)) / (n * (n - 1)))

    return {
        "std_rna": float(zr.std(0).mean()),
        "std_prot": float(zp.std(0).mean()),
        "intra_cos_rna": mean_offdiag_cos(zr),
        "intra_cos_prot": mean_offdiag_cos(zp),
        "pos_cos": float(np.mean(np.sum(Zr * Zp, axis=1))),  # cosinus des paires positives
    }


# --------------------------------------------------------------------------- #
# Scheduler (cosine + warmup optionnel)
# --------------------------------------------------------------------------- #
def make_scheduler(opt, total_steps, warmup):
    def fn(step):
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# --------------------------------------------------------------------------- #
# Boucle d'entraînement
# --------------------------------------------------------------------------- #
def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rna, prot, split = load_views(args.paired_dir)
    tr, va = split == "train", split == "val"
    rna_tr, prot_tr = rna[tr], prot[tr]
    rna_va, prot_va = rna[va], prot[va]
    print(f"device={device} | train={tr.sum()} val={va.sum()} | "
          f"rna_dim={rna.shape[1]} prot_dim={prot.shape[1]}")

    ds = TensorDataset(torch.from_numpy(rna_tr), torch.from_numpy(prot_tr))
    drop_last = len(ds) > args.batch_size
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)

    cfg = CLIPConfig(
    rna_dim=rna.shape[1],
    prot_dim=prot.shape[1],
    dproj=args.dproj,
    depth=args.depth,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    dropout=args.dropout,
    tau=args.tau,)
    model = CLIPModel.from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(dl)) * args.epochs
    sched = make_scheduler(opt, total_steps, args.warmup)
    json.dump(asdict(cfg) | {"lr": args.lr, "weight_decay": args.weight_decay,
                             "batch_size": args.batch_size, "epochs": args.epochs,
                             "warmup": args.warmup, "seed": args.seed},
              open(out / "config.json", "w"), indent=2)

    best_r5, wait, rows = -1.0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = tlr = tlp = 0.0
        nb = 0
        for r, p in dl:
            r, p = r.to(device), p.to(device)
            z_r, z_p = model(r, p)
            loss, l_r2p, l_p2r = info_nce_symmetric(z_r, z_p, model.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            tl += loss.item(); tlr += l_r2p.item(); tlp += l_p2r.item(); nb += 1
        tl, tlr, tlp = tl / nb, tlr / nb, tlp / nb

        # ---- validation ----
        Zr, Zp = embed_all(model, rna_va, prot_va, device)
        rec_rp, medr_rp = retrieval(Zr, Zp)   # ARN → prot
        rec_pr, medr_pr = retrieval(Zp, Zr)   # prot → ARN
        r5 = 0.5 * (rec_rp[5] + rec_pr[5])
        cs = collapse_stats(Zr, Zp)

        row = {
            "epoch": epoch, "lr": opt.param_groups[0]["lr"],
            "train_loss": tl, "train_loss_r2p": tlr, "train_loss_p2r": tlp,
            "val_R1_r2p": rec_rp[1], "val_R5_r2p": rec_rp[5], "val_R10_r2p": rec_rp[10],
            "val_R50_r2p": rec_rp[50], "val_MedR_r2p": medr_rp,
            "val_R1_p2r": rec_pr[1], "val_R5_p2r": rec_pr[5], "val_R10_p2r": rec_pr[10],
            "val_R50_p2r": rec_pr[50], "val_MedR_p2r": medr_pr,
            "val_R5_mean": r5, **cs,
        }
        rows.append(row)
        print(f"[{epoch:3d}/{args.epochs}] loss={tl:.4f} (r→p {tlr:.3f}/p→r {tlp:.3f}) | "
              f"val R@1={100*rec_rp[1]:.2f}/{100*rec_pr[1]:.2f}% R@5={100*rec_rp[5]:.2f}/"
              f"{100*rec_pr[5]:.2f}% MedR={medr_rp:.0f}/{medr_pr:.0f} | "
              f"posCos={cs['pos_cos']:.3f} stdR={cs['std_rna']:.3f} "
              f"intraCos={cs['intra_cos_rna']:.2f}")

        # collapse warning
        if cs["std_rna"] < 1e-3 or cs["intra_cos_rna"] > 0.99:
            print("   [warn] possible collapse (std≈0 ou cosinus intra≈1).")

        # écriture LIVE à chaque époque : tu peux ouvrir metrics.csv et training_curves.png
        # PENDANT l'entraînement et rafraîchir pour suivre la progression en direct.
        _write_csv(out / "metrics.csv", rows)
        _plot_curves(out / "training_curves.png", rows)

        # ---- early stopping sur R@5 val ----
        if r5 > best_r5:
            best_r5 = r5
            wait = 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "epoch": epoch, "val_R5_mean": r5}, out / "best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping (pas d'amélioration de R@5 val depuis {args.patience} époques).")
                break

    torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, out / "last.pt")
    _write_csv(out / "metrics.csv", rows)
    _plot_curves(out / "training_curves.png", rows)
    print(f"\nTerminé en {time.time()-t0:.0f}s. Meilleur R@5 val (moy.) = {best_r5*100:.2f}%.")
    print(f"Sorties dans {out}/ : best.pt, last.pt, metrics.csv, training_curves.png, config.json")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_curves(path, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[info] matplotlib indisponible, courbes non tracées ({e}).")
        return
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ep, [r["train_loss"] for r in rows], label="totale")
    ax[0, 0].plot(ep, [r["train_loss_r2p"] for r in rows], "--", label="ARN→prot")
    ax[0, 0].plot(ep, [r["train_loss_p2r"] for r in rows], "--", label="prot→ARN")
    ax[0, 0].set_title("Loss (train)"); ax[0, 0].set_xlabel("époque"); ax[0, 0].legend()
    for k, c in [("val_R1_r2p", "R@1"), ("val_R5_r2p", "R@5"), ("val_R10_r2p", "R@10"), ("val_R50_r2p", "R@50")]:
        ax[0, 1].plot(ep, [100 * r[k] for r in rows], label=c)
    ax[0, 1].set_title("Recall val (ARN→prot, %)"); ax[0, 1].set_xlabel("époque"); ax[0, 1].legend()
    ax[1, 0].plot(ep, [r["val_MedR_r2p"] for r in rows], label="ARN→prot")
    ax[1, 0].plot(ep, [r["val_MedR_p2r"] for r in rows], label="prot→ARN")
    ax[1, 0].set_title("MedR val (plus bas = mieux)"); ax[1, 0].set_xlabel("époque"); ax[1, 0].legend()
    ax[1, 1].plot(ep, [r["pos_cos"] for r in rows], label="cos paires +")
    ax[1, 1].plot(ep, [r["intra_cos_rna"] for r in rows], label="cos intra ARN")
    ax[1, 1].plot(ep, [r["std_rna"] for r in rows], label="std/dim ARN")
    ax[1, 1].set_title("Alignement & collapse"); ax[1, 1].set_xlabel("époque"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement du CLIP ARN–protéine (baseline).")
    p.add_argument("--paired-dir", required=True, help="Dossier avec paired_rna.npy, paired_protein.npy, paired_cells.csv")
    p.add_argument("--outdir", default="runs/baseline")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--dproj", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=0, help="steps de warmup linéaire (0 = aucun)")
    p.add_argument("--patience", type=int, default=10, help="early stopping (époques sans gain R@5 val)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
