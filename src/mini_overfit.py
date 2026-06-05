"""
Mini-overfit test pour le modèle CLIP ARN-protéine.

Objectif :
    Vérifier que le pipeline peut mémoriser un petit sous-ensemble de cellules.
    Ce n'est PAS une vraie évaluation : on entraîne et on évalue sur les mêmes cellules.

Usage depuis la racine du projet :
    python src/mini_overfit.py \
      --paired-dir MLP/paired \
      --outdir MLP/runs/mini_overfit_n512 \
      --n 512 \
      --epochs 300 \
      --batch-size 512 \
      --lr 1e-3 \
      --dropout 0.0 \
      --weight-decay 0.0 \
      --device auto

Interprétation :
    - Si R@1/R@5 montent très haut sur ce subset : le code/pipeline peut apprendre.
    - Si R@1/R@5 restent bas : suspicion sur l'appariement, l'ordre des cellules,
      la normalisation, la loss, ou un bug dans la construction des paires.
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
    cells = pd.read_csv(d / "paired_cells.csv")
    if "split" not in cells.columns:
        raise ValueError("paired_cells.csv doit contenir une colonne 'split'.")
    split = cells["split"].to_numpy()
    return rna, prot, cells, split


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


def retrieval(Zq, Zg, ks=(1, 5, 10, 50)):
    """
    Rang de la vraie paire dans le même subset.
    Ici, la vraie paire de la ligne i est la colonne i.
    """
    N = len(Zq)
    sims = Zq @ Zg.T
    true = sims[np.arange(N), np.arange(N)][:, None]
    ranks = (sims > true).sum(axis=1) + 1
    rec = {k: float(np.mean(ranks <= k)) for k in ks}
    medr = float(np.median(ranks))
    meanr = float(np.mean(ranks))
    return rec, medr, meanr


def collapse_stats(Zr, Zp):
    def mean_offdiag_cos(z):
        S = z @ z.T
        n = len(z)
        if n <= 1:
            return float("nan")
        return float((S.sum() - np.trace(S)) / (n * (n - 1)))

    return {
        "std_rna": float(Zr.std(0).mean()),
        "std_prot": float(Zp.std(0).mean()),
        "intra_cos_rna": mean_offdiag_cos(Zr),
        "intra_cos_prot": mean_offdiag_cos(Zp),
        "pos_cos": float(np.mean(np.sum(Zr * Zp, axis=1))),
    }


def make_subset(rna, prot, cells, split, n, seed, subset_split):
    if subset_split == "all":
        idx_pool = np.arange(len(rna))
    else:
        idx_pool = np.where(split == subset_split)[0]

    if len(idx_pool) == 0:
        raise ValueError(f"Aucune cellule trouvée pour split == {subset_split!r}.")
    if n > len(idx_pool):
        raise ValueError(f"--n={n} demandé, mais seulement {len(idx_pool)} cellules disponibles.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(idx_pool, size=n, replace=False)
    idx.sort()  # garder un ordre stable et reproductible

    return rna[idx], prot[idx], cells.iloc[idx].copy(), idx


def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rna, prot, cells, split = load_views(args.paired_dir)
    rna_sub, prot_sub, cells_sub, global_idx = make_subset(
        rna, prot, cells, split, args.n, args.seed, args.subset_split
    )

    cells_sub["global_index"] = global_idx
    cells_sub.to_csv(out / "mini_overfit_cells.csv", index=False)

    print("=" * 78)
    print("MINI-OVERFIT TEST")
    print("=" * 78)
    print(f"device={device}")
    print(f"paired_dir={args.paired_dir}")
    print(f"subset_split={args.subset_split} | n={len(rna_sub)}")
    print(f"rna_dim={rna_sub.shape[1]} | prot_dim={prot_sub.shape[1]}")
    print(f"batch_size={args.batch_size} | epochs={args.epochs} | lr={args.lr}")
    print("Important : train == eval sur le même subset. Ce n'est PAS une validation.")
    print("=" * 78)

    ds = TensorDataset(torch.from_numpy(rna_sub), torch.from_numpy(prot_sub))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    cfg = CLIPConfig(
        rna_dim=rna_sub.shape[1],
        prot_dim=prot_sub.shape[1],
        dproj=args.dproj,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        tau=args.tau,
        proj_bias=args.proj_bias,
    )
    model = CLIPModel.from_config(cfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    rows = []
    best_r1 = -1.0
    t0 = time.time()

    # Baseline random attendue sur ce subset
    print(f"Random attendu environ : R@1={100/args.n:.2f}% | R@5={100*min(5,args.n)/args.n:.2f}%")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_r2p = total_p2r = 0.0
        nb = 0

        for r, p in dl:
            r = r.to(device)
            p = p.to(device)

            z_r, z_p = model(r, p)
            loss, l_r2p, l_p2r = info_nce_symmetric(z_r, z_p, model.tau)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += float(loss.item())
            total_r2p += float(l_r2p.item())
            total_p2r += float(l_p2r.item())
            nb += 1

        train_loss = total_loss / max(1, nb)
        train_loss_r2p = total_r2p / max(1, nb)
        train_loss_p2r = total_p2r / max(1, nb)

        should_eval = (epoch == 1) or (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if should_eval:
            Zr, Zp = embed_all(model, rna_sub, prot_sub, device)
            rec_rp, medr_rp, meanr_rp = retrieval(Zr, Zp)
            rec_pr, medr_pr, meanr_pr = retrieval(Zp, Zr)
            cs = collapse_stats(Zr, Zp)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_loss_r2p": train_loss_r2p,
                "train_loss_p2r": train_loss_p2r,
                "R1_r2p": rec_rp[1],
                "R5_r2p": rec_rp[5],
                "R10_r2p": rec_rp[10],
                "R50_r2p": rec_rp[50],
                "MedR_r2p": medr_rp,
                "MeanR_r2p": meanr_rp,
                "R1_p2r": rec_pr[1],
                "R5_p2r": rec_pr[5],
                "R10_p2r": rec_pr[10],
                "R50_p2r": rec_pr[50],
                "MedR_p2r": medr_pr,
                "MeanR_p2r": meanr_pr,
                **cs,
            }
            rows.append(row)

            r1_mean = 0.5 * (rec_rp[1] + rec_pr[1])
            r5_mean = 0.5 * (rec_rp[5] + rec_pr[5])
            if r1_mean > best_r1:
                best_r1 = r1_mean
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "epoch": epoch,
                        "R1_mean": r1_mean,
                        "R5_mean": r5_mean,
                    },
                    out / "best.pt",
                )

            print(
                f"[{epoch:4d}/{args.epochs}] "
                f"loss={train_loss:.4f} "
                f"(r→p {train_loss_r2p:.3f}/p→r {train_loss_p2r:.3f}) | "
                f"R@1={100*rec_rp[1]:.1f}/{100*rec_pr[1]:.1f}% "
                f"R@5={100*rec_rp[5]:.1f}/{100*rec_pr[5]:.1f}% "
                f"MedR={medr_rp:.0f}/{medr_pr:.0f} | "
                f"posCos={cs['pos_cos']:.3f} "
                f"intraCos={cs['intra_cos_rna']:.3f}/{cs['intra_cos_prot']:.3f}"
            )

            if args.stop_r1 is not None and r1_mean >= args.stop_r1:
                print(f"Arrêt : R@1 moyen >= {100*args.stop_r1:.1f}% atteint.")
                break

    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
        },
        out / "last.pt",
    )

    if rows:
        with open(out / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(out / "config.json", "w") as f:
        json.dump(
            {
                **asdict(cfg),
                "paired_dir": args.paired_dir,
                "outdir": args.outdir,
                "n": args.n,
                "subset_split": args.subset_split,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "device": str(device),
            },
            f,
            indent=2,
        )

    print("=" * 78)
    print(f"Terminé en {time.time() - t0:.0f}s.")
    print(f"Sorties : {out}/best.pt, last.pt, metrics.csv, mini_overfit_cells.csv, config.json")
    print("=" * 78)


def parse_args():
    p = argparse.ArgumentParser(description="Mini-overfit test CLIP ARN-protéine.")
    p.add_argument("--paired-dir", required=True)
    p.add_argument("--outdir", default="MLP/runs/mini_overfit")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--subset-split", default="train", choices=["train", "val", "test", "all"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--dproj", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--proj-bias", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--stop-r1", type=float, default=0.95, help="Arrêt si R@1 moyen atteint cette valeur. Mettre -1 pour désactiver.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = p.parse_args()

    if args.stop_r1 is not None and args.stop_r1 < 0:
        args.stop_r1 = None
    if args.n < 2:
        raise ValueError("--n doit être >= 2.")
    if args.batch_size < 2:
        raise ValueError("--batch-size doit être >= 2.")
    return args


if __name__ == "__main__":
    train(parse_args())
