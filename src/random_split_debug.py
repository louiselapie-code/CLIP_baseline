"""
Random split debug pour le CLIP ARN-protéine.

Objectif : distinguer un problème de code/signal d'un problème de généralisation spatiale.

Par défaut, le script NE TOUCHE PAS au vrai split test :
  - il prend uniquement les cellules dont paired_cells.csv contient split == "train" ;
  - il recrée à l'intérieur un train/val random ;
  - il entraîne sur le train random ;
  - il évalue sur la val random.

Usage conseillé depuis la racine du projet :

python src/random_split_debug.py \
  --paired-dir MLP/paired \
  --outdir MLP/runs/random_split_debug_trainpool \
  --pool-splits train \
  --val-frac 0.15 \
  --epochs 50 \
  --batch-size 1024 \
  --tau 0.05 \
  --lr 3e-4 \
  --device auto

Interprétation :
  - random split >> spatial split  : le problème vient surtout du spatial holdout / distribution shift.
  - random split aussi mauvais     : problème de signal, preprocessing, architecture, ou appariement plus profond.
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
# Device / données
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
    cells = pd.read_csv(d / "paired_cells.csv")
    if "split" not in cells.columns:
        raise ValueError("paired_cells.csv doit contenir une colonne 'split'.")
    if len(rna) != len(prot) or len(rna) != len(cells):
        raise ValueError(
            f"Tailles incohérentes : rna={len(rna)}, prot={len(prot)}, cells={len(cells)}"
        )
    return rna, prot, cells


def make_random_split(cells: pd.DataFrame, pool_splits: str, val_frac: float, seed: int):
    requested = [s.strip() for s in pool_splits.split(",") if s.strip()]
    if requested == ["all"]:
        pool_mask = np.ones(len(cells), dtype=bool)
    else:
        pool_mask = cells["split"].astype(str).isin(requested).to_numpy()

    pool_idx = np.flatnonzero(pool_mask)
    if len(pool_idx) < 10:
        raise ValueError(f"Pool trop petit pour pool_splits={pool_splits!r} : {len(pool_idx)} cellules")

    rng = np.random.default_rng(seed)
    shuffled = pool_idx.copy()
    rng.shuffle(shuffled)

    n_val = int(round(len(shuffled) * val_frac))
    n_val = max(1, min(n_val, len(shuffled) - 1))

    val_idx = np.sort(shuffled[:n_val])
    train_idx = np.sort(shuffled[n_val:])
    return train_idx, val_idx, pool_idx


def maybe_subsample(idx: np.ndarray, max_n: int | None, seed: int) -> np.ndarray:
    if max_n is None or max_n <= 0 or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_n, replace=False))


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_subset(model, rna, prot, idx, device, bs=8192):
    model.eval()
    zr, zp = [], []
    for s in range(0, len(idx), bs):
        batch_idx = idx[s:s + bs]
        r = torch.from_numpy(rna[batch_idx]).to(device)
        p = torch.from_numpy(prot[batch_idx]).to(device)
        a, b = model(r, p)
        zr.append(a.float().cpu().numpy())
        zp.append(b.float().cpu().numpy())
    return np.concatenate(zr), np.concatenate(zp)


def retrieval(Zq, Zg, ks=(1, 5, 10, 50), chunk=1024):
    """Rang de la vraie paire, en supposant que Zq[i] correspond à Zg[i]."""
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
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Zr), size=min(sample, len(Zr)), replace=False)
    zr, zp = Zr[idx], Zp[idx]

    def mean_offdiag_cos(z):
        if len(z) < 2:
            return float("nan")
        S = z @ z.T
        n = len(z)
        return float((S.sum() - np.trace(S)) / (n * (n - 1)))

    return {
        "std_rna": float(zr.std(0).mean()),
        "std_prot": float(zp.std(0).mean()),
        "intra_cos_rna": mean_offdiag_cos(zr),
        "intra_cos_prot": mean_offdiag_cos(zp),
        "pos_cos": float(np.mean(np.sum(Zr * Zp, axis=1))),
    }


def evaluate(model, rna, prot, idx, device, tag: str, seed: int):
    Zr, Zp = embed_subset(model, rna, prot, idx, device)
    rec_rp, medr_rp = retrieval(Zr, Zp)
    rec_pr, medr_pr = retrieval(Zp, Zr)
    cs = collapse_stats(Zr, Zp, seed=seed)
    out = {
        f"{tag}_n": len(idx),
        f"{tag}_R1_r2p": rec_rp[1],
        f"{tag}_R5_r2p": rec_rp[5],
        f"{tag}_R10_r2p": rec_rp[10],
        f"{tag}_R50_r2p": rec_rp[50],
        f"{tag}_MedR_r2p": medr_rp,
        f"{tag}_R1_p2r": rec_pr[1],
        f"{tag}_R5_p2r": rec_pr[5],
        f"{tag}_R10_p2r": rec_pr[10],
        f"{tag}_R50_p2r": rec_pr[50],
        f"{tag}_MedR_p2r": medr_pr,
        f"{tag}_R5_mean": 0.5 * (rec_rp[5] + rec_pr[5]),
    }
    out.update({f"{tag}_{k}": v for k, v in cs.items()})
    return out


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def make_scheduler(opt, total_steps, warmup):
    def fn(step):
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# --------------------------------------------------------------------------- #
# Entraînement random split debug
# --------------------------------------------------------------------------- #
def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rna, prot, cells = load_views(args.paired_dir)
    train_idx, val_idx, pool_idx = make_random_split(cells, args.pool_splits, args.val_frac, args.seed)

    # Sous-échantillonnages optionnels utiles pour tests rapides.
    train_idx = maybe_subsample(train_idx, args.max_train, args.seed + 1)
    val_idx = maybe_subsample(val_idx, args.max_val, args.seed + 2)
    train_eval_idx = maybe_subsample(train_idx, args.train_eval_n, args.seed + 3)

    split_df = cells.copy()
    split_df["random_debug_split"] = "unused"
    split_df.loc[train_idx, "random_debug_split"] = "train_random"
    split_df.loc[val_idx, "random_debug_split"] = "val_random"
    split_df.to_csv(out / "random_debug_cells.csv", index=False)

    ds = TensorDataset(torch.from_numpy(rna[train_idx]), torch.from_numpy(prot[train_idx]))
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
        tau=args.tau,
        proj_bias=args.proj_bias,
    )
    model = CLIPModel.from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(dl)) * args.epochs
    sched = make_scheduler(opt, total_steps, args.warmup)

    config_out = asdict(cfg) | {
        "paired_dir": args.paired_dir,
        "pool_splits": args.pool_splits,
        "pool_n": int(len(pool_idx)),
        "train_random_n": int(len(train_idx)),
        "val_random_n": int(len(val_idx)),
        "train_eval_n": int(len(train_eval_idx)),
        "val_frac": args.val_frac,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup": args.warmup,
        "seed": args.seed,
        "device": str(device),
    }
    with open(out / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print("=" * 78)
    print("RANDOM SPLIT DEBUG")
    print("=" * 78)
    print(f"device={device}")
    print(f"paired_dir={args.paired_dir}")
    print(f"pool_splits={args.pool_splits} | pool_n={len(pool_idx)}")
    print(f"train_random={len(train_idx)} | val_random={len(val_idx)} | train_eval={len(train_eval_idx)}")
    print(f"rna_dim={rna.shape[1]} | prot_dim={prot.shape[1]}")
    print(f"batch_size={args.batch_size} | epochs={args.epochs} | lr={args.lr} | tau={args.tau}")
    print("Important : split random interne au pool. Ce n'est PAS une évaluation finale.")
    print("=" * 78)
    print(
        f"Random attendu sur val : R@1≈{100/len(val_idx):.4f}% | "
        f"R@5≈{100*5/len(val_idx):.4f}%"
    )

    rows = []
    best_r5 = -1.0
    wait = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = tlr = tlp = 0.0
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
            sched.step()
            tl += loss.item()
            tlr += l_r2p.item()
            tlp += l_p2r.item()
            nb += 1

        row = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_loss": tl / max(1, nb),
            "train_loss_r2p": tlr / max(1, nb),
            "train_loss_p2r": tlp / max(1, nb),
        }

        do_eval = (epoch == 1) or (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            val_metrics = evaluate(model, rna, prot, val_idx, device, "val_random", args.seed)
            train_metrics = evaluate(model, rna, prot, train_eval_idx, device, "train_random", args.seed)
            row.update(val_metrics)
            row.update(train_metrics)

            r5 = val_metrics["val_random_R5_mean"]
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"loss={row['train_loss']:.4f} "
                f"(r→p {row['train_loss_r2p']:.3f}/p→r {row['train_loss_p2r']:.3f}) | "
                f"VAL R@1={100*val_metrics['val_random_R1_r2p']:.2f}/"
                f"{100*val_metrics['val_random_R1_p2r']:.2f}% "
                f"R@5={100*val_metrics['val_random_R5_r2p']:.2f}/"
                f"{100*val_metrics['val_random_R5_p2r']:.2f}% "
                f"MedR={val_metrics['val_random_MedR_r2p']:.0f}/"
                f"{val_metrics['val_random_MedR_p2r']:.0f} | "
                f"TRAIN(sample) R@5={100*train_metrics['train_random_R5_r2p']:.2f}/"
                f"{100*train_metrics['train_random_R5_p2r']:.2f}% | "
                f"posCos={val_metrics['val_random_pos_cos']:.3f} "
                f"intraCos={val_metrics['val_random_intra_cos_rna']:.3f}/"
                f"{val_metrics['val_random_intra_cos_prot']:.3f}"
            )

            if r5 > best_r5:
                best_r5 = r5
                wait = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "epoch": epoch,
                        "val_random_R5_mean": r5,
                    },
                    out / "best.pt",
                )
            else:
                wait += 1
                if wait >= args.patience:
                    print(f"Early stopping : pas de gain de R@5 val_random depuis {args.patience} évaluations.")
                    rows.append(row)
                    break
        else:
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"loss={row['train_loss']:.4f} "
                f"(r→p {row['train_loss_r2p']:.3f}/p→r {row['train_loss_p2r']:.3f})"
            )

        rows.append(row)

    torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, out / "last.pt")
    write_csv(out / "metrics.csv", rows)
    print("=" * 78)
    print(f"Terminé en {time.time() - t0:.0f}s.")
    print(f"Meilleur R@5 val_random moyen = {100 * best_r5:.2f}%")
    print(f"Sorties : {out}/best.pt, last.pt, metrics.csv, random_debug_cells.csv, config.json")
    print("=" * 78)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    # Union des clés, car les lignes sans évaluation n'ont pas toutes les colonnes.
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Random split debug pour CLIP ARN-protéine.")
    p.add_argument("--paired-dir", required=True, help="Dossier avec paired_rna.npy, paired_protein.npy, paired_cells.csv")
    p.add_argument("--outdir", default="MLP/runs/random_split_debug")

    p.add_argument("--pool-splits", default="train", help="Splits originaux à utiliser comme pool, ex: train ou train,val ou all")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-train", type=int, default=0, help="0 = tout garder ; sinon sous-échantillonne le train random")
    p.add_argument("--max-val", type=int, default=0, help="0 = tout garder ; sinon sous-échantillonne la val random")
    p.add_argument("--train-eval-n", type=int, default=5000, help="Nombre max de cellules train utilisées pour le retrieval train")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--dproj", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--proj-bias", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--patience", type=int, default=8, help="Nombre d'évaluations sans gain avant stop")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
