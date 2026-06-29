#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figures de l'annotation scConcept (pour la présentation), par slide.

Lit les SORTIES de tools/scconcept_annotate.py :
  - data/processed/<slide>/scconcept_labels.csv    (cell_id, split, scconcept_leiden)
  - data/processed/<slide>/scconcept_embedding.npy (N×D, MÊME ordre que le csv)
et, pour la carte spatiale + le crosstab :
  - data/raw/<slide>/h5ad/<slide>_protein_with_spatial_split_seed42.h5ad  (obs x_um/y_um)
  - data/raw/<slide>/tables/<slide>_celltype_annotations_seed42.csv        (col qc_celltype_cpu)

Produit (dans --outdir, défaut <root>/outputs/figs) :
  1. scconcept_<slide>_umap_spatial.png  -> UMAP des embeddings scConcept + carte spatiale, couleur = cluster Leiden
  2. scconcept_<slide>_cluster_sizes.png -> nombre de cellules par cluster
  3. scconcept_<slide>_crosstab.png      -> heatmap cluster scConcept vs ancien qc_celltype_cpu (si dispo)

Usage (depuis la racine du projet, là où est data/) :
    python make_scconcept_viz.py --slide cosmx_breast
    python make_scconcept_viz.py --slide all --root $HOME/CLIP_baseline

Test rapide SANS données (vérifie juste que les figures se génèrent) :
    python make_scconcept_viz.py --demo

Dépendances : numpy, pandas, matplotlib, scikit-learn (+ anndata pour la carte
spatiale ; umap-learn optionnel — sinon repli automatique sur une PCA 2D).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SLIDES = ["cosmx_breast", "xenium_renal"]
CLUSTER_COL = "scconcept_leiden"   # colonne écrite par scconcept_annotate.py
OLD_COL = "qc_celltype_cpu"        # ancienne annotation (pour le crosstab de contrôle)


# --------------------------------------------------------------------------- #
# Palette : assez de couleurs distinctes pour ~40 clusters
# --------------------------------------------------------------------------- #
def make_palette(keys):
    base = []
    for name in ("tab20", "tab20b", "tab20c"):
        base += list(plt.get_cmap(name).colors)
    return {k: base[i % len(base)] for i, k in enumerate(keys)}


def sort_clusters(values):
    """Trie les labels de cluster numériquement si possible ('0','1',...,'10')."""
    vals = [str(v) for v in pd.unique(np.asarray(values, dtype=str))]
    try:
        return sorted(vals, key=lambda s: int(s))
    except ValueError:
        return sorted(vals)


# --------------------------------------------------------------------------- #
# Réduction 2D : UMAP si dispo, sinon PCA
# --------------------------------------------------------------------------- #
def embed2d(X, method="auto", seed=42):
    """Réduction 2D pour l'affichage : UMAP si dispo (plus joli), sinon PCA (numpy/SVD)."""
    if method in ("auto", "umap"):
        try:
            import umap
            return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed).fit_transform(X), "UMAP"
        except Exception as e:
            if method == "umap":
                raise
            print(f"   [info] UMAP indisponible ({e}) -> PCA")
    # PCA 2D via SVD — aucune dépendance externe (numpy seul)
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, "PCA"


# --------------------------------------------------------------------------- #
# Coordonnées spatiales depuis le .h5ad, réalignées sur l'ordre `ids`
# --------------------------------------------------------------------------- #
def load_coords(h5ad_path, ids):
    if not Path(h5ad_path).exists():
        return None
    import anndata as ad
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs
    if {"x_um", "y_um"}.issubset(obs.columns):
        xy = np.column_stack([obs["x_um"].to_numpy(float), obs["y_um"].to_numpy(float)])
    elif "spatial" in a.obsm:                       # repli : coordonnées dans obsm['spatial']
        xy = np.asarray(a.obsm["spatial"])[:, :2].astype(float)
    else:
        return None
    candidates = [np.asarray(a.obs_names, dtype=str)]
    if "cell_id" in obs.columns:
        candidates.append(obs["cell_id"].astype(str).to_numpy())
    for key in candidates:
        m = {k: i for i, k in enumerate(key)}
        hit = np.fromiter((c in m for c in ids), dtype=bool, count=len(ids))
        if hit.mean() > 0.9:
            out = np.full((len(ids), 2), np.nan)
            idx = np.array([m.get(c, -1) for c in ids])
            ok = idx >= 0
            out[ok] = xy[idx[ok]]
            print(f"   [coords] {hit.mean() * 100:.0f}% des cellules localisées")
            return out
    print("   [info] cell_id non appariables au h5ad -> pas de carte spatiale")
    return None


def load_old_labels(annot_path, ids):
    if not Path(annot_path).exists():
        return None
    df = pd.read_csv(annot_path)
    if OLD_COL not in df.columns:
        return None
    idc = next((c for c in df.columns if c.lower() in ("cell_id", "cellid")), df.columns[0])
    mp = dict(zip(df[idc].astype(str), df[OLD_COL].astype(str)))
    return pd.Series(ids).map(mp).to_numpy()


# --------------------------------------------------------------------------- #
# FIGURE 1 : UMAP + carte spatiale, couleur = cluster
# --------------------------------------------------------------------------- #
def fig_umap_spatial(slide, emb, clusters, coords, outdir, n_umap, n_spatial, method, seed):
    clu = np.asarray(clusters, dtype=str)
    order = sort_clusters(clu)
    sizes = pd.Series(clu).value_counts()
    color = make_palette(order)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # --- UMAP des embeddings scConcept ---
    iu = rng.choice(len(emb), size=min(n_umap, len(emb)), replace=False)
    XY, meth = embed2d(np.asarray(emb[iu]), method=method, seed=seed)
    for t in order:
        m = clu[iu] == t
        if m.any():
            ax[0].scatter(XY[m, 0], XY[m, 1], s=6, alpha=0.6, color=color[t],
                          label=f"{t} (n={int(sizes.get(t, 0)):,})")
    ax[0].set_title(f"{meth} des embeddings scConcept — {slide}\n(échantillon {len(iu):,} cellules)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].legend(title="cluster", markerscale=2.2, fontsize=7,
                 loc="best", frameon=True, ncol=1 if len(order) <= 14 else 2)

    # --- carte spatiale ---
    if coords is not None:
        isp = rng.choice(len(emb), size=min(n_spatial, len(emb)), replace=False)
        for t in order:
            m = (clu[isp] == t) & np.isfinite(coords[isp, 0])
            if m.any():
                ax[1].scatter(coords[isp][m, 0], coords[isp][m, 1], s=2, alpha=0.5, color=color[t])
        ax[1].set_aspect("equal"); ax[1].invert_yaxis()
        ax[1].set_title(f"Carte spatiale (x_um, y_um) — {slide}\n(échantillon {min(n_spatial, len(emb)):,} cellules)")
        ax[1].set_xlabel("x (µm)"); ax[1].set_ylabel("y (µm)")
    else:
        ax[1].text(0.5, 0.5, "coordonnées spatiales indisponibles", ha="center", va="center")
        ax[1].set_axis_off()

    fig.suptitle(f"Annotation scConcept (Leiden) — {slide} : {len(order)} clusters, {len(clu):,} cellules",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = outdir / f"scconcept_{slide}_umap_spatial.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    print(f"   -> {fn}")
    return fn


# --------------------------------------------------------------------------- #
# FIGURE 2 : tailles de clusters
# --------------------------------------------------------------------------- #
def fig_cluster_sizes(slide, clusters, outdir):
    clu = pd.Series(np.asarray(clusters, dtype=str))
    order = sort_clusters(clu)
    sizes = clu.value_counts().reindex(order).fillna(0).astype(int)
    color = make_palette(order)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(order) + 2), 4.5))
    bars = ax.bar(range(len(order)), sizes.values, color=[color[t] for t in order])
    tot = int(sizes.sum())
    for b, v in zip(bars, sizes.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{100 * v / tot:.0f}%",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=8)
    ax.set_xlabel("cluster scConcept (Leiden)"); ax.set_ylabel("nombre de cellules")
    ax.set_ylim(0, sizes.max() * 1.18)
    ax.set_title(f"Taille des clusters — {slide} ({tot:,} cellules)")
    fig.tight_layout()
    fn = outdir / f"scconcept_{slide}_cluster_sizes.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    print(f"   -> {fn}")
    return fn


# --------------------------------------------------------------------------- #
# FIGURE 3 : crosstab cluster scConcept vs ancien qc_celltype_cpu
# --------------------------------------------------------------------------- #
def fig_crosstab(slide, clusters, old, outdir):
    if old is None:
        print("   [info] pas d'annotation qc_celltype_cpu -> pas de crosstab")
        return None
    clu = pd.Series(np.asarray(clusters, dtype=str), name="scConcept").reset_index(drop=True)
    old = pd.Series(np.asarray(old, dtype=str), name=OLD_COL).reset_index(drop=True)
    keep = ~old.isin(["nan", "None", "NA", "NaN", ""])
    clu, old = clu[keep], old[keep]
    if clu.empty:
        print("   [info] crosstab vide -> ignoré")
        return None
    ct = pd.crosstab(clu, old).reindex(sort_clusters(clu.values))
    ctn = ct.div(ct.sum(1), axis=0)  # chaque cluster -> sa composition en anciens types (ligne = 1)

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * ct.shape[1] + 3), max(4, 0.42 * ct.shape[0] + 2)))
    im = ax.imshow(ctn.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index, fontsize=8)
    ax.set_xlabel(f"ancien type ({OLD_COL})"); ax.set_ylabel("cluster scConcept")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ctn.values[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="fraction du cluster")
    ax.set_title(f"Correspondance scConcept ↔ ancienne annotation — {slide}\n(chaque ligne somme à 1)")
    fig.tight_layout()
    fn = outdir / f"scconcept_{slide}_crosstab.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    print(f"   -> {fn}")
    return fn


# --------------------------------------------------------------------------- #
def run_slide(slide, root, outdir, h5ad, annot, n_umap, n_spatial, method, seed):
    pdir = root / "data" / "processed" / slide
    labels_csv = pdir / "scconcept_labels.csv"
    emb_npy = pdir / "scconcept_embedding.npy"
    if not labels_csv.exists() or not emb_npy.exists():
        print(f"[{slide}] SORTIES MANQUANTES : {labels_csv} et/ou {emb_npy}\n"
              f"          (lance d'abord tools/scconcept_annotate.py)")
        return
    print(f"\n[{slide}] lecture des sorties scConcept")
    cells = pd.read_csv(labels_csv)
    ids = cells["cell_id"].astype(str).to_numpy()
    clusters = cells[CLUSTER_COL].astype(str).to_numpy()
    emb = np.load(emb_npy, mmap_mode="r")
    assert emb.shape[0] == len(ids), f"désaccord embedding ({emb.shape[0]}) / labels ({len(ids)})"
    print(f"   {len(ids):,} cellules, {pd.unique(clusters).size} clusters, embedding {tuple(emb.shape)}")

    if h5ad:
        cand_h5 = [Path(h5ad)]
    else:                                            # RNA d'abord (cell_id identiques à scConcept), puis protéine
        h5dir = root / "data" / "raw" / slide / "h5ad"
        cand_h5 = [h5dir / f"{slide}_rna_with_spatial_split_seed42.h5ad",
                   h5dir / f"{slide}_protein_with_spatial_split_seed42.h5ad"]
    coords = None
    for h5 in cand_h5:
        coords = load_coords(h5, ids)
        if coords is not None:
            break
    if coords is None:
        print("   [info] coordonnées spatiales introuvables -> carte spatiale ignorée")
    an = Path(annot) if annot else root / "data" / "raw" / slide / "tables" / f"{slide}_celltype_annotations_seed42.csv"
    old = load_old_labels(an, ids)

    fig_umap_spatial(slide, emb, clusters, coords, outdir, n_umap, n_spatial, method, seed)
    fig_cluster_sizes(slide, clusters, outdir)
    fig_crosstab(slide, clusters, old, outdir)


def run_demo(outdir, seed=42):
    print("=== DEMO (données fictives, pas de vraies sorties scConcept) ===")
    rng = np.random.default_rng(seed)
    n, k = 4000, 5
    centers = rng.normal(0, 6, size=(k, 40))
    z = rng.integers(0, k, size=n)
    emb = centers[z] + rng.normal(0, 1, size=(n, 40))
    clusters = z.astype(str)
    coords = np.column_stack([rng.uniform(0, 2000, n), rng.uniform(0, 1500, n)])
    coords[z == 0] += np.array([1500, 0])      # un domaine spatial décalé, pour voir de la structure
    old_types = np.array(["Tumor", "Stroma", "Immune", "Endothelial"])
    old = old_types[np.clip(z, 0, 3)]
    fig_umap_spatial("demo", emb, clusters, coords, outdir, 4000, 4000, "pca", seed)
    fig_cluster_sizes("demo", clusters, outdir)
    fig_crosstab("demo", clusters, old, outdir)
    print("DEMO OK")


def main():
    ap = argparse.ArgumentParser(description="Figures de l'annotation scConcept (par slide).")
    ap.add_argument("--slide", default="all", help="cosmx_breast | xenium_renal | all")
    ap.add_argument("--root", default=".", help="racine du projet (où est data/). Défaut: dossier courant.")
    ap.add_argument("--outdir", default=None, help="dossier de sortie (défaut <root>/outputs/figs)")
    ap.add_argument("--h5ad", default=None, help="override du .h5ad pour les coordonnées (sinon déduit du slide)")
    ap.add_argument("--annot", default=None, help="override du CSV qc_celltype_cpu (sinon déduit du slide)")
    ap.add_argument("--n-umap", type=int, default=15000, help="nb de cellules échantillonnées pour l'UMAP")
    ap.add_argument("--n-spatial", type=int, default=120000, help="nb de cellules échantillonnées pour la carte spatiale")
    ap.add_argument("--method", default="auto", choices=["auto", "umap", "pca"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="génère des figures de test sans vraies données")
    a = ap.parse_args()

    root = Path(a.root).expanduser()
    outdir = Path(a.outdir).expanduser() if a.outdir else root / "outputs" / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Sortie -> {outdir}")

    if a.demo:
        run_demo(outdir, a.seed)
        return

    slides = SLIDES if a.slide == "all" else [a.slide]
    for s in slides:
        run_slide(s, root, outdir, a.h5ad, a.annot, a.n_umap, a.n_spatial, a.method, a.seed)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
