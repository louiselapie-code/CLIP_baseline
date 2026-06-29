"""
run_cellular_neighborhoods.py — Baseline de niches PROTÉIQUES dédiée (Cellular Neighborhoods).

Méthode standard de la protéomique spatiale (Schürch et al., Cell 2020, CODEX) :
  1. phénotyper les cellules par la PROTÉINE  (KMeans sur les marqueurs)
  2. pour chaque cellule, composition de son VOISINAGE spatial en phénotypes (fenêtre kNN)
  3. KMeans sur ces compositions  → niches/domaines

C'est la "bonne" baseline protéique (≠ BANKSY générique sur intensités). Entièrement protéine,
sans torch. Sortie : domains_cn_protein.npy aligné sur --paired-dir → src/pathology_validation.py.

Exemple :
  python benchmark/run_cellular_neighborhoods.py \
    --paired-dir data/processed/xenium_renal \
    --spatial-h5ad data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
    --n-domains 5 --n-phenotypes 15 --k 20 --outdir eval/benchmark_xenium/protein_cn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_niches import load_views, load_obsm_aligned  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Cellular Neighborhoods sur la protéine.")
    ap.add_argument("--paired-dir", required=True, help="protéine = paired_protein.npy")
    ap.add_argument("--spatial-h5ad", required=True, help="pour obsm['spatial_um']")
    ap.add_argument("--spatial-obsm", default="spatial_um")
    ap.add_argument("--n-domains", type=int, default=5)
    ap.add_argument("--n-phenotypes", type=int, default=15, help="nb de phénotypes protéiques (étape 1)")
    ap.add_argument("--k", type=int, default=20, help="voisins de la fenêtre (étape 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="eval/benchmark_xenium/protein_cn")
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    _, prot, cell_id, _ = load_views(a.paired_dir)
    coords, _ = load_obsm_aligned(a.spatial_h5ad, a.spatial_obsm, cell_id, ndim=2)
    print(f"Cellular Neighborhoods : {len(prot)} cellules × {prot.shape[1]} marqueurs "
          f"| {a.n_phenotypes} phénotypes, fenêtre k={a.k}")

    from scipy.sparse import csr_matrix
    from sklearn.cluster import KMeans
    from sklearn.neighbors import kneighbors_graph

    # 1) phénotypes protéiques
    P = (prot - prot.mean(0)) / (prot.std(0) + 1e-8)
    pheno = KMeans(n_clusters=a.n_phenotypes, random_state=a.seed, n_init="auto").fit_predict(P)
    onehot = csr_matrix((np.ones(len(pheno)), (np.arange(len(pheno)), pheno)),
                        shape=(len(pheno), a.n_phenotypes))
    # 2) composition du voisinage en phénotypes
    A = kneighbors_graph(np.asarray(coords, float), n_neighbors=a.k, mode="connectivity")
    A = A.multiply(1.0 / A.sum(1))
    comp = np.asarray(A @ onehot.toarray())  # (N, n_phenotypes)
    # 3) niches = clusters de compositions
    dom = KMeans(n_clusters=a.n_domains, random_state=a.seed, n_init="auto").fit_predict(comp)
    np.save(out / "domains_cn_protein.npy", dom.astype(np.int64))
    print(f"écrit {out/'domains_cn_protein.npy'} ({a.n_domains} domaines)")


if __name__ == "__main__":
    main()
