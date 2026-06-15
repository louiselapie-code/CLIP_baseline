#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Le signal fin intra-type existe-t-il, et quel encodeur ARN le porte le mieux ?

Retrieval cross-modal restreint à UN type (galerie = un seul type), vs hasard.
Côté ARN on peut comparer plusieurs encodeurs, SANS rien ré-entraîner :
  - NOVAE  (paired_rna.npy, niche-level) -> via le CLIP entraîné (--ckpt) ET la CCA.
  - scConcept (obsm d'un .h5ad, cell-level) -> via la CCA seulement (le CLIP est lié à
    NOVAE ; on ne peut pas y faire passer scConcept sans ré-entraîner).

Comparer CCA(NOVAE) vs CCA(scConcept) dit lequel des deux encodeurs ARN contient le plus
de signal cellule-à-cellule corrélé à la protéine — donc s'il vaut la peine de ré-entraîner
le CLIP sur scConcept.

Usage :
  python eval_intra_type.py --paired-dir data/processed/cosmx_breast \
      --ckpt runs/clip_cosmx_seed42/best.pt \
      --labels annotation_out/labels_cosmx.csv --label-col cell_type \
      --rna-h5ad annotation_out/cosmx_annotated.h5ad --rna-obsm X_scConcept \
      --min-cells 200 --device cpu
"""
import argparse
import numpy as np
import pandas as pd

from evaluate import load_views, load_model, embed, l2   # briques validées de ton repo
from sanity_check import cca_fit, cca_transform


def retrieval_at_k(A, B, ks=(1, 5, 10)):
    """A[i]<->B[i] appariés ; galerie = lignes de B ; rang de la vraie paire (diagonale)."""
    A = l2(np.asarray(A, np.float32)); B = l2(np.asarray(B, np.float32))
    S = A @ B.T
    diag = np.diag(S).copy()
    ranks = (S > diag[:, None]).sum(1) + 1
    return {k: float((ranks <= k).mean()) for k in ks}, float(np.median(ranks))


def intra_type_table(Zr, Zp, labels, name, min_cells=200, max_cells=5000, seed=0, ks=(1, 5, 10)):
    rng = np.random.default_rng(seed); rows = []
    for t in pd.unique(labels):
        idx = np.where(labels == t)[0]; n = len(idx)
        if n < min_cells:
            continue
        if n > max_cells:
            idx = rng.choice(idx, max_cells, replace=False); n = max_cells
        rec_ab, medr_ab = retrieval_at_k(Zr[idx], Zp[idx], ks)
        floor = 1.0 / n
        rows.append({"espace": name, "type": str(t), "N": n,
                     "R@1_AB": round(100 * rec_ab[1], 3), "R@5_AB": round(100 * rec_ab[5], 3),
                     "MedR_AB": medr_ab, "floor_R@1%": round(100 * floor, 3),
                     "ratio_R@1": round(rec_ab[1] / floor, 1) if floor > 0 else np.nan,
                     "MedR/(N/2)": round(medr_ab / (n / 2), 2)})
    return pd.DataFrame(rows)


def load_obsm_aligned(h5ad, key, cell_id):
    """Charge obsm[key] d'un .h5ad et l'aligne sur l'ordre des cellules appariées (par cell_id)."""
    import anndata as ad
    a = ad.read_h5ad(h5ad)
    if key not in a.obsm:
        raise KeyError(f"obsm['{key}'] absent de {h5ad}. Disponibles : {list(a.obsm)}")
    emb = np.asarray(a.obsm[key], dtype=np.float32)
    idx = {str(c): i for i, c in enumerate(a.obs["cell_id"].astype(str))}
    rows = np.array([idx.get(str(c), -1) for c in cell_id])
    found = rows >= 0
    out = np.zeros((len(cell_id), emb.shape[1]), dtype=np.float32)
    out[found] = emb[rows[found]]
    print(f"[{key}] {found.sum()}/{len(cell_id)} cellules appariées trouvées ({found.mean()*100:.1f}%)")
    return out, found


def cca_intra(rna_src, prot, tr, te, lab, keep, found, name, comps, mc, xc):
    """CCA(rna_src, prot) ajustée sur train, puis retrieval intra-type sur test."""
    trv, tev = tr & found, te & found
    cca = cca_fit(rna_src[trv], prot[trv], comps)
    A, B = cca_transform(cca, rna_src[tev], prot[tev])
    m = keep[tev]
    return intra_type_table(A[m], B[m], lab[tev][m], name, mc, xc)


def main():
    p = argparse.ArgumentParser(description="Signal fin intra-type + comparaison d'encodeurs ARN.")
    p.add_argument("--paired-dir", required=True)
    p.add_argument("--ckpt", default=None, help="checkpoint CLIP (NOVAE) ; optionnel")
    p.add_argument("--labels", required=True)
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--rna-h5ad", default=None, help=".h5ad avec un embedding ARN alternatif (ex: scConcept)")
    p.add_argument("--rna-obsm", default="X_scConcept", help="clé obsm de l'embedding alternatif")
    p.add_argument("--min-cells", type=int, default=200)
    p.add_argument("--max-cells", type=int, default=5000)
    p.add_argument("--cca-components", type=int, default=32)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    import torch
    device = torch.device(a.device)
    rna, prot, cell_id, split = load_views(a.paired_dir)
    tr, te = split == "train", split == "test"
    allf = np.ones(len(cell_id), dtype=bool)

    df = pd.read_csv(a.labels)
    idc = "cell_id" if "cell_id" in df.columns else df.columns[0]
    lc = a.label_col if a.label_col in df.columns else df.columns[1]
    mp = dict(zip(df[idc].astype(str), df[lc].astype(str)))
    lab = np.array([mp.get(c, "NA") for c in cell_id], dtype=object)
    keep = lab != "NA"

    tabs = []
    # CLIP entraîné (NOVAE) si checkpoint fourni
    if a.ckpt:
        model = load_model(a.ckpt, device)
        Zr_te, Zp_te = embed(model, rna[te], prot[te], device)
        m = keep[te]
        tabs.append(intra_type_table(Zr_te[m], Zp_te[m], lab[te][m], "CLIP (NOVAE)", a.min_cells, a.max_cells))

    # CCA sur NOVAE brut
    tabs.append(cca_intra(rna, prot, tr, te, lab, keep, allf, "CCA (NOVAE)", a.cca_components, a.min_cells, a.max_cells))

    # CCA sur l'encodeur ARN alternatif (scConcept)
    if a.rna_h5ad:
        emb_alt, found = load_obsm_aligned(a.rna_h5ad, a.rna_obsm, cell_id)
        tabs.append(cca_intra(emb_alt, prot, tr, te, lab, keep, found,
                              f"CCA ({a.rna_obsm})", a.cca_components, a.min_cells, a.max_cells))

    out = pd.concat(tabs, ignore_index=True).sort_values(["type", "espace"])
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Retrieval INTRA-TYPE — comparaison d'encodeurs ARN ===")
    print(out.to_string(index=False))
    print("\nLecture : pour chaque type, compare 'CCA (NOVAE)' et 'CCA (X_scConcept)'.")
    print("  ratio_R@1 plus haut / MedR/(N/2) plus bas  =>  cet encodeur ARN porte plus de signal fin")
    print("  => si scConcept gagne nettement, ré-entraîne le CLIP avec X_scConcept en entrée ARN.")


if __name__ == "__main__":
    main()
