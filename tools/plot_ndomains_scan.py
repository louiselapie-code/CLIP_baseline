"""
plot_ndomains_scan.py — Agrège les résultats du scan n_domains et trace ARI/NMI vs n.

Lit eval/scan_<dataset>/patho_n<n>/pathology_summary.csv pour chaque n, et produit :
  scan_summary.csv   (méthode × n : ARI, NMI, homogénéité)
  scan_ndomains.png  (2 courbes : ARI vs n et NMI vs n, une ligne par méthode)

Usage : python tools/plot_ndomains_scan.py --base eval/scan_xenium --ns "5 7 10 15"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ns", default="5 7 10 15")
    a = ap.parse_args()
    base = Path(a.base)
    ns = [int(x) for x in a.ns.split()]

    rows = []
    for n in ns:
        f = base / f"patho_n{n}" / "pathology_summary.csv"
        if not f.exists():
            print(f"  [manque] {f}")
            continue
        df = pd.read_csv(f)
        df["n_domains"] = n
        rows.append(df)
    if not rows:
        print("Aucun pathology_summary.csv trouvé."); return
    allr = pd.concat(rows, ignore_index=True)
    allr.to_csv(base / "scan_summary.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for met, sub in allr.groupby("méthode"):
        sub = sub.sort_values("n_domains")
        ax[0].plot(sub["n_domains"], sub["ARI_vs_patho"], "o-", label=met)
        ax[1].plot(sub["n_domains"], sub["NMI_vs_patho"], "o-", label=met)
    for k, lab in [(0, "ARI"), (1, "NMI")]:
        ax[k].set_xlabel("n_domains"); ax[k].set_ylabel(f"{lab} vs pathologie")
        ax[k].set_title(f"{lab} ↔ pathologie selon n_domains"); ax[k].grid(alpha=0.3); ax[k].legend()
    fig.suptitle("Robustesse : accord niches ↔ pathologie en fonction du nombre de domaines",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(base / "scan_ndomains.png", dpi=140)
    plt.close(fig)
    print(f"écrit {base/'scan_ndomains.png'} et scan_summary.csv\n")
    print("NMI vs pathologie :")
    print(allr.pivot_table(index="méthode", columns="n_domains", values="NMI_vs_patho").to_string())


if __name__ == "__main__":
    main()
