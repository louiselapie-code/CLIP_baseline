"""
run_banksy.py — Baseline de domaines spatiaux « BANKSY-style » (ARN seul), pour le benchmark.

BANKSY (Singhal et al., Nat Genet 2024) augmente chaque cellule par : (1) sa propre
expression et (2) la moyenne de l'expression de son voisinage spatial, pondérées par λ
(λ grand = domaines tissulaires). On reproduit ici cette idée cœur :

    features = [ sqrt(1-λ)·z(PCA propre) , sqrt(λ)·z(PCA moyenné sur le voisinage) ]
    → KMeans(n_domains)

⚠️ Version transparente SANS le filtre de Gabor azimutal (AGF) du package officiel
`banksy_py`. Pour un résultat « officiel », installe banksy_py. Mais cette baseline capture
l'essentiel (propre + voisinage, λ-pondéré) et tourne sans torch.

Sortie : domains_banksy.npy (aligné sur l'ordre des cellules de --paired-dir) → se branche
directement sur src/pathology_validation.py --maps.

Exemple :
  python benchmark/run_banksy.py \
    --rna-h5ad data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
    --paired-dir data/processed/xenium_renal --n-domains 5 --outdir results/eval/benchmark_xenium
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_niches import load_views, load_obsm_aligned  # noqa: E402


def banksy_features(counts_sparse, coords, n_pcs=20, k=18, lam=0.8, seed=0):
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import kneighbors_graph

    X = counts_sparse.copy()
    X.data = np.log1p(X.data)
    Z = TruncatedSVD(n_components=min(n_pcs, X.shape[1] - 1), random_state=seed).fit_transform(X)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)  # z-score
    # moyenne du voisinage spatial (matrice kNN normalisée par ligne @ Z)
    A = kneighbors_graph(np.asarray(coords, float), n_neighbors=k, mode="connectivity")
    A = A.multiply(1.0 / A.sum(1))            # normalisation par ligne
    Znbr = np.asarray(A @ Z)
    Znbr = (Znbr - Znbr.mean(0)) / (Znbr.std(0) + 1e-8)
    return np.hstack([np.sqrt(1 - lam) * Z, np.sqrt(lam) * Znbr])


def main():
    ap = argparse.ArgumentParser(description="Baseline BANKSY-style (ARN seul).")
    ap.add_argument("--rna-h5ad", required=True, help="h5ad de comptages bruts (X)")
    ap.add_argument("--paired-dir", required=True, help="pour l'ordre des cell_id (alignement)")
    ap.add_argument("--spatial-obsm", default="spatial_um")
    ap.add_argument("--n-domains", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.8, help="λ BANKSY (grand = domaines)")
    ap.add_argument("--k", type=int, default=18, help="voisins spatiaux")
    ap.add_argument("--n-pcs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/eval/benchmark_xenium")
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    _, _, cell_id, _ = load_views(a.paired_dir)
    coords, _ = load_obsm_aligned(a.rna_h5ad, a.spatial_obsm, cell_id, ndim=2)

    # comptages alignés sur les cellules appariées
    import anndata as ad
    import scipy.sparse as sp
    adata = ad.read_h5ad(a.rna_h5ad, backed="r")
    ids = adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs.columns else adata.obs.index.astype(str)
    pos = {str(c): i for i, c in enumerate(ids)}
    rows = np.array([pos[str(c)] for c in cell_id])
    X = adata.X[rows]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    print(f"BANKSY-style : {X.shape[0]} cellules × {X.shape[1]} gènes | λ={a.lam} k={a.k}")

    F = banksy_features(X, coords, n_pcs=a.n_pcs, k=a.k, lam=a.lam, seed=a.seed)
    from sklearn.cluster import KMeans
    dom = KMeans(n_clusters=a.n_domains, random_state=a.seed, n_init="auto").fit_predict(F)
    np.save(out / "domains_banksy.npy", dom.astype(np.int64))
    print(f"écrit {out/'domains_banksy.npy'} ({a.n_domains} domaines)")


if __name__ == "__main__":
    main()
