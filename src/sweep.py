"""
Recherche légère d'hyperparamètres (§3 étape 3, §5) pour le CLIP ARN–protéine.

Balaie une grille (produit cartésien) sur les 3 leviers prioritaires — τ, learning rate,
taille de batch — en lançant train.py pour chaque config, puis classe les runs par
**Recall@5 val** (le critère d'early stopping). Écrit un résumé et désigne le meilleur run.

Usage (valeurs par défaut = grille légère) :
    python sweep.py --paired-dir MLP/paired --outdir MLP/runs/sweep1

Personnaliser la grille :
    python sweep.py --paired-dir MLP/paired --outdir MLP/runs/sweep1 \
        --taus 0.05 0.07 0.1 --lrs 1e-4 3e-4 --batch-sizes 512 1024 --epochs 30

Chaque config produit son propre sous-dossier (best.pt, metrics.csv, courbes). Le meilleur
best.pt est ensuite à passer à evaluate.py.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from itertools import product
from pathlib import Path

TRAIN = Path(__file__).resolve().parent / "train.py"


def best_r5_from_csv(metrics_csv: Path):
    """Renvoie (meilleur val_R5_mean, époque) depuis le metrics.csv d'un run, ou (None, None)."""
    if not metrics_csv.exists():
        return None, None
    best, best_ep = -1.0, None
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            r5 = float(row["val_R5_mean"])
            if r5 > best:
                best, best_ep = r5, int(row["epoch"])
    return (best, best_ep) if best >= 0 else (None, None)


def run_one(cfg: dict, paired_dir: str, outdir: Path, epochs: int, patience: int,
            device: str, seed: int) -> dict:
    tag = f"tau{cfg['tau']}_lr{cfg['lr']}_bs{cfg['batch_size']}"
    run_dir = outdir / tag
    cmd = [
        sys.executable, str(TRAIN),
        "--paired-dir", paired_dir, "--outdir", str(run_dir),
        "--tau", str(cfg["tau"]), "--lr", str(cfg["lr"]),
        "--batch-size", str(cfg["batch_size"]),
        "--epochs", str(epochs), "--patience", str(patience),
        "--device", device, "--seed", str(seed),
    ]
    print(f"\n{'='*70}\n▶ RUN {tag}\n{'='*70}")
    ret = subprocess.run(cmd)  # stdout hérité → tu vois la progression de chaque run
    best, ep = best_r5_from_csv(run_dir / "metrics.csv")
    return {**cfg, "tag": tag, "best_val_R5_mean": best, "best_epoch": ep,
            "ok": ret.returncode == 0 and best is not None, "best_ckpt": str(run_dir / "best.pt")}


def main():
    p = argparse.ArgumentParser(description="Sweep léger τ / LR / batch pour le CLIP.")
    p.add_argument("--paired-dir", required=True)
    p.add_argument("--outdir", default="results/runs/sweep")
    p.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.07, 0.1])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1024])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid = [{"tau": t, "lr": lr, "batch_size": bs}
            for t, lr, bs in product(a.taus, a.lrs, a.batch_sizes)]
    print(f"Grille : {len(grid)} configs "
          f"(τ={a.taus} × lr={a.lrs} × batch={a.batch_sizes}), {a.epochs} époques max chacune.")

    results = [run_one(c, a.paired_dir, outdir, a.epochs, a.patience, a.device, a.seed) for c in grid]

    # Classement par R@5 val décroissant
    ranked = sorted(results, key=lambda r: (r["best_val_R5_mean"] is None, -(r["best_val_R5_mean"] or -1)))
    with open(outdir / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "tau", "lr", "batch_size",
                                          "best_val_R5_mean", "best_epoch", "ok", "best_ckpt"])
        w.writeheader()
        w.writerows(ranked)

    print(f"\n{'='*70}\nRÉSUMÉ (trié par R@5 val décroissant)\n{'='*70}")
    print(f"{'config':<28}{'R@5 val':>10}{'époque':>9}")
    for r in ranked:
        r5 = f"{r['best_val_R5_mean']*100:.2f}%" if r["best_val_R5_mean"] is not None else "ÉCHEC"
        print(f"{r['tag']:<28}{r5:>10}{str(r['best_epoch']):>9}")
    best = ranked[0]
    if best["best_val_R5_mean"] is not None:
        print(f"\n★ Meilleure config : {best['tag']}  (R@5 val = {best['best_val_R5_mean']*100:.2f}%)")
        print(f"   checkpoint : {best['best_ckpt']}")
        print(f"   → à passer à : python evaluate.py --paired-dir {a.paired_dir} "
              f"--ckpt {best['best_ckpt']} --outdir <eval_dir>")
    print(f"\nRésumé écrit : {outdir/'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
