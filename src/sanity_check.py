"""
Sanity check du couplage ARN↔protéine + baselines de retrieval cross-modal.

But (avant tout entraînement) :
  1. valider que le couplage par cell_id est correct (alignement, splits, exclusion des nuls) ;
  2. vérifier qu'il existe un signal partagé RNA↔protéine *récupérable linéairement* — sinon,
     soit le couplage est faux, soit les deux modalités sont quasi indépendantes (et le CLIP
     n'a aucune chance). C'est le "préalable" du §8 (récupération croisée).

Baselines (cf. §8.2) :
  - **Aléatoire** : plancher théorique (Recall@k = k/N, MedR ≈ N/2). Référence.
  - **CCA** (Canonical Correlation Analysis) : apprend un espace linéaire COMMUN aux deux
    modalités (ajusté sur le TRAIN), où l'on peut comparer ARN et protéine par cosinus.
    NB : une PCA *indépendante* par modalité ne convient pas — ses axes ne sont pas comparables
    d'une modalité à l'autre, donc un cosinus inter-modalités serait au niveau du hasard. CCA est
    l'analogue linéaire correct du CLIP.

Métriques : Recall@1/5/10 et MedR (rang médian du bon appariement), dans les deux directions
(ARN→protéine et protéine→ARN), évaluées sur val et test (galerie = toutes les cellules du split).

Usage :
    python sanity_check.py \
        --rna-npy  novae_embeddings/cosmx_breast_rna_..._novae_latent.npy \
        --rna-csv  novae_embeddings/cosmx_breast_rna_..._cells.csv \
        --prot-npy protein_features/cosmx_breast_protein_..._protein_features.npy \
        --prot-csv protein_features/cosmx_breast_protein_..._cells.csv \
        --save paired            # (optionnel) sauve les vues alignées pour l'entraînement
"""
from __future__ import annotations

import argparse

import numpy as np

from paired_data import couple, save_paired


# --------------------------------------------------------------------------- #
# Retrieval cross-modal
# --------------------------------------------------------------------------- #
def cross_modal_retrieval(Q: np.ndarray, G: np.ndarray, ks=(1, 5, 10, 50), chunk: int = 2048):
    """Pour chaque requête Q[i], rang de la vraie galerie G[i] par similarité cosinus.

    Q et G sont alignés (G[i] = bon appariement de Q[i]) et dans le MÊME espace.
    """
    eps = 1e-12
    Qn = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), eps, None)
    Gn = G / np.clip(np.linalg.norm(G, axis=1, keepdims=True), eps, None)
    N = len(Q)
    ranks = np.empty(N, dtype=np.int64)
    for s in range(0, N, chunk):  # par blocs pour ne pas matérialiser N×N
        e = min(s + chunk, N)
        sims = Qn[s:e] @ Gn.T                      # (e-s, N)
        true_sim = sims[np.arange(e - s), np.arange(s, e)][:, None]
        ranks[s:e] = (sims > true_sim).sum(axis=1) + 1   # 1-based
    recall = {k: float(np.mean(ranks <= k)) for k in ks}
    return recall, float(np.median(ranks)), N


def _fmt(recall, medr, N):
    rs = "  ".join(f"R@{k}={v*100:6.2f}%" for k, v in recall.items())
    return f"{rs}  MedR={medr:6.0f} / {N}"


def report_retrieval(name: str, A: np.ndarray, B: np.ndarray):
    """A = ARN (espace commun), B = protéine (espace commun)."""
    r_ab, m_ab, N = cross_modal_retrieval(A, B)
    r_ba, m_ba, _ = cross_modal_retrieval(B, A)
    print(f"  [{name}]  galerie N={N}")
    print(f"     ARN→prot : {_fmt(r_ab, m_ab, N)}")
    print(f"     prot→ARN : {_fmt(r_ba, m_ba, N)}")
    return r_ab, r_ba


def random_floor(N: int, ks=(1, 5, 10, 50)):
    rec = {k: k / N for k in ks}
    return rec, (N + 1) / 2.0


def sanitize_finite(data):
    """Remplace NaN/Inf par 0 (sinon SVD/retrieval plantent) en signalant la cause."""
    for name, M in [("ARN", data.rna), ("protéine", data.protein)]:
        bad = ~np.isfinite(M)
        if bad.any():
            rows = int(np.any(bad, axis=1).sum())
            cols = int(np.any(bad, axis=0).sum())
            print(f"[warn] {name} : {int(bad.sum())} valeurs non finies (NaN/Inf) "
                  f"sur {rows} cellules et {cols} dim(s)/marqueur(s) → remplacées par 0.")
            if cols and rows > 0.5 * len(M):
                print(f"       ⚠️ {cols} dim(s) touchent >50% des cellules : probablement "
                      f"des NaN à la SOURCE (preprocessing). À corriger en amont.")
            M[bad] = 0.0
    return data


# --------------------------------------------------------------------------- #
# CCA — implémentation close-form stable (pas de scikit-learn).
# On blanchit chaque vue puis SVD de la covariance croisée (matrice 64×64 → SVD
# qui converge toujours). Une régularisation ridge gère la collinéarité / le rang
# faible des embeddings, là où sklearn.CCA (NIPALS) plante ("SVD did not converge").
# --------------------------------------------------------------------------- #
def _inv_sqrt(C: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, floor, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def cca_fit(X: np.ndarray, Y: np.ndarray, k: int, reg: float = 1e-3) -> dict:
    """CCA close-form ajustée sur (X, Y). Standardise chaque dim, blanchit, SVD."""
    eps = 1e-8
    mux, sdx = X.mean(0), np.where(X.std(0) < eps, 1.0, X.std(0))
    muy, sdy = Y.mean(0), np.where(Y.std(0) < eps, 1.0, Y.std(0))
    Xs, Ys = (X - mux) / sdx, (Y - muy) / sdy
    n, p, q = Xs.shape[0], Xs.shape[1], Ys.shape[1]
    Cxx = Xs.T @ Xs / n + reg * np.eye(p)
    Cyy = Ys.T @ Ys / n + reg * np.eye(q)
    Cxy = Xs.T @ Ys / n
    Wx, Wy = _inv_sqrt(Cxx), _inv_sqrt(Cyy)
    U, s, Vt = np.linalg.svd(Wx @ Cxy @ Wy, full_matrices=False)
    k = min(k, len(s))
    return dict(mux=mux, sdx=sdx, muy=muy, sdy=sdy,
                A=Wx @ U[:, :k], B=Wy @ Vt[:k].T, corr=s[:k])


def cca_transform(params: dict, X: np.ndarray, Y: np.ndarray):
    # Pondération par la corrélation canonique : les dimensions à faible corrélation
    # (bruit) sont atténuées dans le cosinus, ce qui rend la baseline plus juste.
    Xs = (X - params["mux"]) / params["sdx"]
    Ys = (Y - params["muy"]) / params["sdy"]
    w = params["corr"]
    return (Xs @ params["A"]) * w, (Ys @ params["B"]) * w


def cca_baseline(data, n_components: int, splits=("val", "test")):
    rtr, ptr, _ = data.subset("train")
    nc = min(n_components, rtr.shape[1], ptr.shape[1])
    print(f"\n=== Baseline CCA (n_components={nc}, ajustée sur train: {len(rtr)} cellules) ===")
    params = cca_fit(rtr, ptr, nc)
    print(f"  corrélations canoniques (top 5, 1≈signal fort / 0≈aucun) : "
          f"{np.round(params['corr'][:5], 3).tolist()}")
    for sp in splits:
        rv, pv, _ = data.subset(sp)
        if len(rv) == 0:
            continue
        A, B = cca_transform(params, rv, pv)
        report_retrieval(f"CCA {sp}", A, B)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Sanity check + baselines retrieval cross-modal.")
    p.add_argument("--rna-npy", required=True)
    p.add_argument("--rna-csv", required=True)
    p.add_argument("--prot-npy", required=True)
    p.add_argument("--prot-csv", required=True)
    p.add_argument("--id-col", default="cell_id")
    p.add_argument("--split-col", default="split")
    p.add_argument("--cca-components", type=int, default=32)
    p.add_argument("--save", default=None, help="Dossier où sauver les vues alignées (optionnel).")
    a = p.parse_args()

    data = couple(a.rna_npy, a.rna_csv, a.prot_npy, a.prot_csv,
                  id_col=a.id_col, split_col=a.split_col, drop_null_rna=True)
    data = sanitize_finite(data)
    if a.save:
        save_paired(data, a.save)

    print("\n=== Plancher aléatoire (référence) ===")
    for sp in ("val", "test"):
        sub = data.subset(sp)[0]
        if len(sub) == 0:
            continue
        N = len(sub)
        rec, medr = random_floor(N)
        print(f"  [aléatoire {sp}]  galerie N={N}")
        print(f"     {_fmt(rec, medr, N)}")

    cca_baseline(data, a.cca_components)

    print("\nLecture : si la CCA bat NETTEMENT le plancher aléatoire (R@k plus élevé, MedR "
          "bien plus petit que N/2), le couplage est bon et il y a du signal ARN↔protéine "
          "→ le CLIP a une chance d'apprendre. Sinon, revérifier l'appariement cell_id.")


if __name__ == "__main__":
    main()
