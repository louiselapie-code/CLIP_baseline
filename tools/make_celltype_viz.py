"""
Visualisation des types cellulaires (Leiden qc_celltype_cpu) par slide :
  - UMAP des embeddings NOVAE (vue ARN), couleur = type cellulaire ;
  - carte spatiale (x_um, y_um), couleur = type cellulaire.
But : vérifier la cohérence des annotations (séparation en UMAP + domaines spatiaux).

Usage : python make_celltype_viz.py --slide cosmx_breast   (ou xenium_renal)
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("/sessions/keen-adoring-galileo/mnt/CLIP_baseline_v0")
OUT = Path("/sessions/keen-adoring-galileo/mnt/outputs/figs"); OUT.mkdir(parents=True, exist_ok=True)
PROT_H5AD = {
  "cosmx_breast": "data/raw/cosmx_breast/h5ad/cosmx_breast_protein_with_spatial_split_seed42.h5ad",
  "xenium_renal": "data/raw/xenium_renal/h5ad/xenium_renal_protein_with_spatial_split_seed42.h5ad",
}
ANNOT = {
  "cosmx_breast": "data/raw/cosmx_breast/tables/cosmx_breast_celltype_annotations_seed42.csv",
  "xenium_renal": "data/raw/xenium_renal/tables/xenium_renal_celltype_annotations_seed42.csv",
}


def coords_map(slide, ids):
    import anndata as ad
    a = ad.read_h5ad(P / PROT_H5AD[slide], backed="r")
    obs = a.obs
    xy_cols = [c for c in ("x_um", "y_um") if c in obs.columns]
    if len(xy_cols) < 2:
        return None
    df = pd.DataFrame({"x": obs["x_um"].to_numpy(), "y": obs["y_um"].to_numpy()})
    # essaie obs_names puis colonne cell_id
    for key in [np.asarray(a.obs_names, dtype=str),
                (obs["cell_id"].astype(str).to_numpy() if "cell_id" in obs.columns else None)]:
        if key is None:
            continue
        m = dict(zip(key, zip(df["x"], df["y"])))
        hit = np.array([c in m for c in ids])
        if hit.mean() > 0.9:
            xy = np.array([m.get(c, (np.nan, np.nan)) for c in ids], dtype=float)
            return xy
    return None


def embed2d(X, seed=42):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed).fit_transform(X), "UMAP"
    except Exception as e:
        from sklearn.decomposition import PCA
        print(f"[info] UMAP indispo ({e}) -> PCA"); return PCA(n_components=2, random_state=seed).fit_transform(X), "PCA"


def main(slide, n_umap=12000, n_spatial=120000, seed=42):
    pdir = P / "data/processed" / slide
    rna = np.load(pdir / "paired_rna.npy").astype(np.float32)
    cells = pd.read_csv(pdir / "paired_cells.csv")
    ids = cells["cell_id"].astype(str).to_numpy()
    an = pd.read_csv(P / ANNOT[slide])
    idc = next((c for c in an.columns if c.lower() in ("cell_id", "cellid")), an.columns[0])
    lab = pd.Series(ids).map(dict(zip(an[idc].astype(str), an["qc_celltype_cpu"].astype(str)))).to_numpy()
    keep = pd.notna(lab)
    rna, ids, lab = rna[keep], ids[keep], lab[keep].astype(str)
    types = sorted(set(lab.tolist()))
    cmap = plt.get_cmap("tab10" if len(types) <= 10 else "tab20")
    color = {t: cmap(i % cmap.N) for i, t in enumerate(types)}
    rng = np.random.default_rng(seed)

    xy = coords_map(slide, ids)
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # --- UMAP (ARN/NOVAE) ---
    iu = rng.choice(len(rna), size=min(n_umap, len(rna)), replace=False)
    XY, meth = embed2d(rna[iu])
    for t in types:
        m = lab[iu] == t
        ax[0].scatter(XY[m, 0], XY[m, 1], s=5, alpha=0.6, color=color[t], label=f"{t} (n={int((lab==t).sum())})")
    ax[0].set_title(f"{meth} des embeddings NOVAE (ARN) — {slide}\n(échantillon {len(iu)} cellules)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].legend(markerscale=2.5, fontsize=8, loc="best", frameon=True)

    # --- carte spatiale ---
    if xy is not None:
        isp = rng.choice(len(rna), size=min(n_spatial, len(rna)), replace=False)
        for t in types:
            m = (lab[isp] == t) & np.isfinite(xy[isp, 0])
            ax[1].scatter(xy[isp][m, 0], xy[isp][m, 1], s=2, alpha=0.5, color=color[t])
        ax[1].set_aspect("equal"); ax[1].invert_yaxis()
        ax[1].set_title(f"Carte spatiale (x_um, y_um) — {slide}\n(échantillon {len(isp)} cellules)")
        ax[1].set_xlabel("x (µm)"); ax[1].set_ylabel("y (µm)")
    else:
        ax[1].text(0.5, 0.5, "coordonnées x_um/y_um introuvables", ha="center", va="center")
        ax[1].set_axis_off()

    fig.suptitle(f"Types cellulaires (Leiden qc_celltype_cpu) — {slide} : {len(types)} types, {keep.sum()} cellules", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = OUT / f"celltype_{slide}.png"
    fig.savefig(fn, dpi=130); plt.close(fig)
    print(f"OK -> {fn}  ({meth}; spatial={'oui' if xy is not None else 'non'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True, choices=["cosmx_breast", "xenium_renal"])
    ap.add_argument("--n-umap", type=int, default=12000)
    ap.add_argument("--n-spatial", type=int, default=120000)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
