"""
Diagnostic radius NOVAE ↔ buffer : risque de fuite spatiale entre splits.

Deux analyses sur un .h5ad ARN (coordonnées spatiales en µm + colonne `split`) :

1) DISTRIBUTION DES LONGUEURS D'ARÊTES du graphe spatial.
   NOVAE construit un graphe de Delaunay puis coupe les arêtes plus longues que `radius`
   (ici on a utilisé radius=100 µm). On reconstruit ce même graphe de Delaunay (scipy) et on
   regarde la distribution des longueurs : ça dit quel radius garde les vrais voisins tout en
   coupant les arêtes aberrantes (longues, aux bords/trous du tissu). → radius proposé.

2) LARGEUR DE BANDE DU BUFFER vs CHAMP RÉCEPTIF.
   NOVAE a calculé les embeddings sur la slide COMPLÈTE (buffer inclus), donc les cellules
   buffer servent de PONT entre splits. Pour qu'une cellule train ne « voie » pas une cellule
   val/test, la bande buffer doit être plus large que le champ réceptif de NOVAE
   ≈ (nb de couches de message-passing) × radius. On mesure la largeur de bande comme la
   distance minimale entre une cellule train et la cellule val/test la plus proche (KDTree) :
   c'est exactement le trou physique occupé par le buffer. On compare à L×radius pour
   plusieurs L (le nombre exact de couches de NOVAE est À CONFIRMER dans sa config).

Sortie : rapport console + PNG (2 panneaux).

Usage :
    python diagnose_radius_buffer.py \
        --h5ad data/raw/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
        --radius 100 --layers 1 2 3 4 --outdir runs/diag
    # Xenium en pixels : ajouter --scale 0.2125 pour convertir en µm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def delaunay_edge_lengths(coords: np.ndarray) -> np.ndarray:
    """Longueurs des arêtes du graphe de Delaunay (le même que NOVAE avant coupe par radius)."""
    from scipy.spatial import Delaunay
    tri = Delaunay(coords)
    s = tri.simplices
    edges = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)


def nearest_cross_split(coords, mask_a, mask_b):
    """Pour chaque cellule de A, distance à la cellule de B la plus proche (KDTree)."""
    from scipy.spatial import cKDTree
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return np.array([])
    tree = cKDTree(coords[mask_b])
    d, _ = tree.query(coords[mask_a], k=1)
    return d


def main():
    p = argparse.ArgumentParser(description="Diagnostic radius NOVAE ↔ buffer (fuite spatiale).")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--radius", type=float, default=100.0, help="radius utilisé par NOVAE (µm)")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="nombres de couches de message-passing à tester (NOVAE : à confirmer)")
    p.add_argument("--split-col", default="split")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--scale", type=float, default=1.0, help="facteur coords→µm (Xenium px : 0.2125)")
    p.add_argument("--propose-pct", type=float, default=99.0, help="percentile pour le radius proposé")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    import anndata as ad
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(a.h5ad)
    coords = np.asarray(adata.obsm[a.spatial_key], dtype=float)[:, :2] * a.scale
    split = adata.obs[a.split_col].astype(str).to_numpy()
    print(f"{adata.n_obs} cellules | splits : "
          f"{ {s: int((split==s).sum()) for s in sorted(set(split))} }")

    # ---- 1) Longueurs d'arêtes ----
    d = delaunay_edge_lengths(coords)
    pcts = [50, 90, 95, 99, 99.9]
    vals = np.percentile(d, pcts)
    prop = float(np.percentile(d, a.propose_pct))
    kept_now = float((d <= a.radius).mean() * 100)
    print("\n=== 1) Longueurs d'arêtes (graphe Delaunay, comme NOVAE) ===")
    print(f"  {len(d):,} arêtes | moyenne={d.mean():.1f} µm | max={d.max():.0f} µm")
    print("  percentiles (µm) : " + "  ".join(f"p{p}={v:.1f}" for p, v in zip(pcts, vals)))
    print(f"  radius ACTUEL = {a.radius:.0f} µm → garde {kept_now:.2f}% des arêtes")
    print(f"  radius PROPOSÉ (p{a.propose_pct}) = {prop:.1f} µm → garde {a.propose_pct:.0f}%")
    if a.radius > 3 * vals[1]:  # radius >> p90
        print(f"  → le radius actuel ({a.radius:.0f}) est très au-dessus du p90 ({vals[1]:.1f}) : "
              "beaucoup d'arêtes longues/aberrantes conservées (cohérent avec le warning NOVAE).")

    # ---- 2) Buffer vs champ réceptif ----
    is_train = split == "train"
    is_eval = np.isin(split, ["val", "test"])
    dte = nearest_cross_split(coords, is_train, is_eval)
    print("\n=== 2) Largeur de bande buffer vs champ réceptif ===")
    if len(dte) == 0:
        print("  [!] pas de cellules train et/ou val/test trouvées — vérifie --split-col.")
        gap = None
    else:
        gap = float(dte.min())
        gp = np.percentile(dte, [0.1, 1, 5, 50])
        print(f"  distance train→(val/test) la plus proche : MIN={gap:.1f} µm "
              f"(p0.1={gp[0]:.1f}, p1={gp[1]:.1f}, p5={gp[2]:.1f}, médiane={gp[3]:.0f})")
        print("  (le MIN ≈ l'endroit le plus fin de la bande buffer = le pire cas)")
        print("\n  Verdict fuite (bande doit être > champ réceptif = couches × radius) :")
        print(f"  {'couches L':>9} | {'RF@radius='+str(int(a.radius)):>16} | "
              f"{'RF@proposé='+f'{prop:.0f}':>16} | radius max sûr (=MIN/L)")
        for L in a.layers:
            rf_now = L * a.radius
            rf_prop = L * prop
            safe_now = "OK" if gap >= rf_now else "⚠️ FUITE"
            safe_prop = "OK" if gap >= rf_prop else "⚠️ FUITE"
            print(f"  {L:>9} | {rf_now:>10.0f} µm {safe_now:>5} | "
                  f"{rf_prop:>10.0f} µm {safe_prop:>5} | {gap / L:>8.1f} µm")

    # ---- Recommandation ----
    print("\n=== Recommandation ===")
    if gap is not None:
        worst_L = max(a.layers)
        max_safe_radius = gap / worst_L
        print(f"  Pour L={worst_L} couches, il faut radius ≤ {max_safe_radius:.1f} µm pour éviter la fuite.")
        if max_safe_radius >= vals[1]:   # >= p90 des arêtes
            print(f"  Ce seuil ({max_safe_radius:.1f}) est ≥ p90 des arêtes ({vals[1]:.1f}) : "
                  "tu peux réduire le radius sans fragmenter le graphe. → réduire le radius "
                  "(ex. proposé ci-dessus) puis recalculer les embeddings NOVAE.")
        else:
            print(f"  Ce seuil ({max_safe_radius:.1f}) est < p90 des arêtes ({vals[1]:.1f}) : "
                  "réduire autant fragmenterait le graphe. → la bande buffer est trop fine pour "
                  "ce nb de couches ; mieux vaut ÉLARGIR le buffer (re-split) que réduire le radius.")
        print("  ⚠️ Confirme le nb de couches L de NOVAE (sa config/doc) pour trancher la bonne ligne.")
    print("  Si à L confirmé le verdict est OK partout → pas de fuite, embeddings actuels sains ; "
          "réduire le radius devient une simple amélioration de qualité (optionnelle).")

    _plot(d, dte, a.radius, prop, a.layers, out / "diag_radius_buffer.png")
    print(f"\nFigure : {out/'diag_radius_buffer.png'}")


def _plot(d, dte, radius, prop, layers, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[info] matplotlib indisponible ({e})."); return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    hi = np.percentile(d, 99.5)
    ax[0].hist(d[d <= hi], bins=80)
    ax[0].axvline(radius, color="r", ls="--", label=f"radius actuel {radius:.0f}")
    ax[0].axvline(prop, color="g", ls="--", label=f"proposé {prop:.0f}")
    ax[0].set_title("Longueurs d'arêtes (Delaunay)"); ax[0].set_xlabel("µm"); ax[0].legend()
    if len(dte):
        ax[1].hist(dte, bins=80)
        ax[1].axvline(dte.min(), color="k", ls="-", label=f"MIN bande {dte.min():.0f}")
        for L in layers:
            ax[1].axvline(L * radius, color="r", ls=":", alpha=0.6)
        ax[1].set_title("dist. train→(val/test) ; pointillés rouges = L×radius")
        ax[1].set_xlabel("µm"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
