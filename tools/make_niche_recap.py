"""
make_niche_recap.py — Figures récap (slides) de l'analyse des niches, à partir des CSV
déjà produits dans results/eval/. Aucun calcul lourd, aucune dépendance torch.

Lit :
  results/eval/niches_{cosmx,xenium}/niches_summary.csv
  results/eval/compare_{cosmx,xenium}/compare_summary.csv
  results/eval/info_{cosmx_counts,xenium_counts}/information_summary.csv

Produit dans --outdir (défaut results/eval/recap/) :
  recap1_novae_vs_scconcept.png   FIDE + NMI(vs types) — NOVAE vs scConcept (clip_joint)
  recap2_fide_4espaces.png        FIDE des 4 espaces NOVAE, CosMx vs Xenium
  recap3_information.png          EV protéine (le signal) + EV ARN (près du plancher)

Usage : python tools/make_niche_recap.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"cosmx": "#1f77b4", "xenium": "#ff7f0e",
       "NOVAE": "#2ca02c", "scConcept": "#d62728"}


def _bars(ax, groups, series, title, ylabel, colors, ymax=None, fmt="{:.2f}"):
    """Barres groupées. series = dict label -> liste de valeurs (une par groupe)."""
    n = len(series)
    x = np.arange(len(groups))
    w = 0.8 / n
    for i, (lab, vals) in enumerate(series.items()):
        xi = x + (i - (n - 1) / 2) * w
        bars = ax.bar(xi, vals, w, label=lab, color=colors.get(lab, None), zorder=3)
        for b, v in zip(bars, vals):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=9)


def load_csv(p):
    return pd.read_csv(p) if Path(p).exists() else None


def fig1_compare(base, out):
    cx = load_csv(base / "compare_cosmx/compare_summary.csv")
    xe = load_csv(base / "compare_xenium/compare_summary.csv")
    if cx is None or xe is None:
        print("  [skip] compare_*: CSV manquant"); return
    def val(df, carte, col):
        return float(df.loc[df["carte"] == carte, col].iloc[0])
    groups = ["CosMx", "Xenium"]
    fide = {"NOVAE": [val(cx, "NOVAE", "FIDE"), val(xe, "NOVAE", "FIDE")],
            "scConcept": [val(cx, "scConcept", "FIDE"), val(xe, "scConcept", "FIDE")]}
    nmi = {"NOVAE": [val(cx, "NOVAE", "NMI_types"), val(xe, "NOVAE", "NMI_types")],
           "scConcept": [val(cx, "scConcept", "NMI_types"), val(xe, "scConcept", "NMI_types")]}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    _bars(ax[0], groups, fide, "Continuité spatiale (FIDE) — plus haut = mieux", "FIDE", COL, ymax=0.9)
    _bars(ax[1], groups, nmi, "Recouvrement avec les types cellulaires (NMI)\nplus bas = vraie niche, pas du typage",
          "NMI vs types", COL, ymax=0.65)
    fig.suptitle("Niches multi-omiques (clip_joint) : NOVAE fait des niches spatiales, scConcept du typage",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "recap1_novae_vs_scconcept.png", dpi=150); plt.close(fig)
    print("  recap1_novae_vs_scconcept.png")


def fig2_fide_spaces(base, out):
    cx = load_csv(base / "niches_cosmx/niches_summary.csv")
    xe = load_csv(base / "niches_xenium/niches_summary.csv")
    if cx is None or xe is None:
        print("  [skip] niches_*: CSV manquant"); return
    order = ["novae_raw", "clip_rna", "clip_prot", "clip_joint"]
    def fide(df, sp):
        return float(df.loc[df["espace"] == sp, "FIDE"].iloc[0])
    series = {"CosMx (prot. 64 marqueurs)": [fide(cx, s) for s in order],
              "Xenium (prot. 27 marqueurs)": [fide(xe, s) for s in order]}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    _bars(ax, order, series, "", "FIDE (continuité spatiale)",
          {"CosMx (prot. 64 marqueurs)": COL["cosmx"], "Xenium (prot. 27 marqueurs)": COL["xenium"]},
          ymax=0.9)
    ax.set_title("Le bénéfice multi-omique dépend du panel protéique\n"
                 "CosMx : clip_joint meilleur  •  Xenium : clip_joint < ARN (protéine faible)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "recap2_fide_4espaces.png", dpi=150); plt.close(fig)
    print("  recap2_fide_4espaces.png")


def fig3_information(base, out):
    cx = load_csv(base / "info_cosmx_counts/information_summary.csv")
    xe = load_csv(base / "info_xenium_counts/information_summary.csv")
    if cx is None or xe is None:
        print("  [skip] info_*_counts: CSV manquant"); return
    parts = ["niche_ARN", "niche_prot", "niche_joint"]
    def ev(df, part, col):
        return float(df.loc[df["partition"] == part, col].iloc[0])
    prot = {"CosMx": [ev(cx, p, "EV_protéine") for p in parts],
            "Xenium": [ev(xe, p, "EV_protéine") for p in parts]}
    arn = {"CosMx": [ev(cx, p, "EV_ARN") for p in parts],
           "Xenium": [ev(xe, p, "EV_ARN") for p in parts]}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    _bars(ax[0], parts, prot, "Information PROTÉIQUE (cible indépendante)\nla jointe > niche_ARN = elle intègre du protéique",
          "variance protéique expliquée", {"CosMx": COL["cosmx"], "Xenium": COL["xenium"]}, ymax=0.5)
    _bars(ax[1], parts, arn, "Information ARN (comptages bruts, cible indépendante)\nprès du plancher pour TOUTES : voir note",
          "variance ARN expliquée", {"CosMx": COL["cosmx"], "Xenium": COL["xenium"]}, ymax=0.5, fmt="{:.3f}")
    ax[1].axhline(0.0, color="grey", lw=1)
    ax[1].text(0.5, 0.46, "≈ plancher : une niche de 10 domaines ne peut pas\nrésumer le transcriptome cellule-à-cellule (bruité)",
               ha="center", va="top", fontsize=8, style="italic", transform=ax[1].transData)
    fig.suptitle("Information des niches (axes indépendants) : le signal réel est sur la protéine",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "recap3_information.png", dpi=150); plt.close(fig)
    print("  recap3_information.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--outdir", default="results/eval/recap")
    a = ap.parse_args()
    base = Path(a.eval_dir); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    print("Figures récap ->", out)
    fig1_compare(base, out)
    fig2_fide_spaces(base, out)
    fig3_information(base, out)
    print("Terminé ✓")


if __name__ == "__main__":
    main()
