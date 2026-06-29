"""
Évaluation du CLIP ARN–protéine sur le TEST (§8).

Charge un checkpoint (best.pt) et les vues alignées, puis évalue sur le split test
(jamais vu à l'entraînement) :

  1. Retrieval cross-modal : CLIP vs baselines (aléatoire, CCA) + contrôle négatif permuté
     (R@1/5/10/50 + MedR, les deux directions).
  2. Diagnostics d'embeddings (alignement des paires, collapse).
  3. UMAP (ou PCA si umap-learn absent) des embeddings test, coloré par modalité
     (alignement visuel) et par label si fourni.
  4. Si --labels fourni : clustering (ARI/NMI) et sonde linéaire (accuracy/F1 macro),
     pour le CLIP (joint / ARN / protéine) vs les modalités brutes (NOVAE / protéine).
     → couvre aussi les baselines « mono-modalité » du §8.2.

CosMx n'est pas encore annoté : sans --labels, les parties 1–3 tournent quand même.
Le format attendu de --labels : CSV avec colonnes cell_id, label (ou --label-col).

Usage :
    python evaluate.py --paired-dir MLP/paired --ckpt MLP/runs/sweep1/<best>/best.pt \
        --outdir MLP/eval/baseline [--labels annot.csv --label-col cell_type]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import CLIPConfig, CLIPModel
# Réutilise les fonctions validées du sanity check (retrieval, floor, CCA close-form).
from sanity_check import cca_fit, cca_transform, cross_modal_retrieval, random_floor


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_views(paired_dir):
    d = Path(paired_dir)
    rna = np.load(d / "paired_rna.npy").astype(np.float32)
    prot = np.load(d / "paired_protein.npy").astype(np.float32)
    cells = pd.read_csv(d / "paired_cells.csv")
    return rna, prot, cells["cell_id"].astype(str).to_numpy(), cells["split"].to_numpy()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CLIPModel.from_config(CLIPConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


@torch.no_grad()
def embed(model, rna, prot, device, bs=8192):
    zr, zp = [], []
    for s in range(0, len(rna), bs):
        a, b = model(torch.from_numpy(rna[s:s+bs]).to(device),
                     torch.from_numpy(prot[s:s+bs]).to(device))
        zr.append(a.float().cpu().numpy()); zp.append(b.float().cpu().numpy())
    return np.concatenate(zr), np.concatenate(zp)


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------- #
# Retrieval (un bloc de résultats par méthode)
# --------------------------------------------------------------------------- #
def eval_retrieval(name, A, B, store):
    rec_ab, m_ab, N = cross_modal_retrieval(A, B)
    rec_ba, m_ba, _ = cross_modal_retrieval(B, A)
    store[name] = {"ARN→prot": rec_ab, "MedR_ARN→prot": m_ab,
                   "prot→ARN": rec_ba, "MedR_prot→ARN": m_ba, "N": N}
    print(f"  {name:18s} ARN→prot  R@1/5/10/50 = "
          f"{100*rec_ab[1]:5.2f}/{100*rec_ab[5]:5.2f}/{100*rec_ab[10]:5.2f}/{100*rec_ab[50]:5.2f}%  MedR={m_ab:.0f}")
    print(f"  {'':18s} prot→ARN  R@1/5/10/50 = "
          f"{100*rec_ba[1]:5.2f}/{100*rec_ba[5]:5.2f}/{100*rec_ba[10]:5.2f}/{100*rec_ba[50]:5.2f}%  MedR={m_ba:.0f}")


# --------------------------------------------------------------------------- #
# Diagnostics embeddings
# --------------------------------------------------------------------------- #
def diagnostics(Zr, Zp, sample=4000, seed=0):
    idx = np.random.default_rng(seed).choice(len(Zr), size=min(sample, len(Zr)), replace=False)
    zr, zp = Zr[idx], Zp[idx]
    def offdiag(z):
        S = z @ z.T; n = len(z); return float((S.sum() - np.trace(S)) / (n * (n - 1)))
    return {"pos_cos": float(np.mean(np.sum(Zr * Zp, axis=1))),
            "intra_cos_rna": offdiag(zr), "intra_cos_prot": offdiag(zp),
            "std_dim_rna": float(zr.std(0).mean()), "std_dim_prot": float(zp.std(0).mean())}


# --------------------------------------------------------------------------- #
# Clustering / sonde linéaire (si labels)
# --------------------------------------------------------------------------- #
def cluster_scores(Z, y, k, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    return float(adjusted_rand_score(y, lab)), float(normalized_mutual_info_score(y, lab))


def linear_probe(Ztr, ytr, Zte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Ztr), ytr)
    pred = clf.predict(sc.transform(Zte))
    return float(accuracy_score(yte, pred)), float(f1_score(yte, pred, average="macro"))


# --------------------------------------------------------------------------- #
# UMAP / PCA 2D
# --------------------------------------------------------------------------- #
def embed_2d(X):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X), "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(X), "PCA (installe umap-learn pour mieux)"


def plot_umap(Zr, Zp, outpath, labels=None, sample=4000, seed=0):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[info] matplotlib indisponible, UMAP non tracé ({e})."); return
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Zr), size=min(sample, len(Zr)), replace=False)
    XY, method = embed_2d(np.vstack([Zr[idx], Zp[idx]]))
    n = len(idx); xy_r, xy_p = XY[:n], XY[n:]
    if labels is not None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6.5)); ax = [ax]
    ax[0].scatter(xy_r[:, 0], xy_r[:, 1], s=4, alpha=0.5, label="ARN (z_r)")
    ax[0].scatter(xy_p[:, 0], xy_p[:, 1], s=4, alpha=0.5, label="protéine (z_p)")
    ax[0].set_title(f"{method} — par modalité (recouvrement = alignement)"); ax[0].legend()
    if labels is not None:
        lab = labels[idx]
        cats = sorted(set(lab.tolist()))
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(cats):
            m = lab == c
            ax[1].scatter(xy_r[m, 0], xy_r[m, 1], s=4, alpha=0.6, color=cmap(i % 20), label=str(c))
        ax[1].set_title(f"{method} — ARN coloré par label"); ax[1].legend(markerscale=3, fontsize=7)
    fig.tight_layout(); fig.savefig(outpath, dpi=120); plt.close(fig)
    print(f"  UMAP/PCA sauvé : {outpath}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Évaluation §8 du CLIP sur le test.")
    p.add_argument("--paired-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--outdir", default="results/eval")
    p.add_argument("--labels", default=None, help="CSV cell_id,label (optionnel)")
    p.add_argument("--label-col", default="label")
    p.add_argument("--n-clusters", type=int, default=0, help="0 = nb de labels uniques")
    p.add_argument("--cca-components", type=int, default=32)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if a.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    rna, prot, cell_id, split = load_views(a.paired_dir)
    tr, te = split == "train", split == "test"
    print(f"train={tr.sum()} test={te.sum()} | rna_dim={rna.shape[1]} prot_dim={prot.shape[1]} | device={device}")

    model = load_model(a.ckpt, device)
    Zr_tr, Zp_tr = embed(model, rna[tr], prot[tr], device)
    Zr_te, Zp_te = embed(model, rna[te], prot[te], device)

    report = {"n_train": int(tr.sum()), "n_test": int(te.sum())}

    # ---- 1. Retrieval sur le test ----
    print("\n=== 1. Retrieval cross-modal (TEST) ===")
    store = {}
    N = int(te.sum())
    rec_f, medr_f = random_floor(N)
    print(f"  {'aléatoire (floor)':18s} R@1/5/10/50 = "
          f"{100*rec_f[1]:.3f}/{100*rec_f[5]:.3f}/{100*rec_f[10]:.3f}/{100*rec_f[50]:.3f}%  MedR={medr_f:.0f}  (N={N})")
    store["random_floor"] = {"recall": rec_f, "MedR": medr_f, "N": N}

    cca = cca_fit(rna[tr], prot[tr], a.cca_components)
    A_te, B_te = cca_transform(cca, rna[te], prot[te])
    eval_retrieval("CCA (linéaire)", A_te, B_te, store)
    eval_retrieval("CLIP", Zr_te, Zp_te, store)

    # Contrôle négatif : appariement permuté → doit retomber au niveau du floor.
    perm = np.random.default_rng(0).permutation(N)
    eval_retrieval("CLIP permuté (-)", Zr_te, Zp_te[perm], store)
    report["retrieval"] = store

    # ---- 2. Diagnostics ----
    print("\n=== 2. Diagnostics embeddings (TEST) ===")
    diag = diagnostics(Zr_te, Zp_te)
    print(f"  cos paires+ = {diag['pos_cos']:.3f} | cos intra ARN/prot = "
          f"{diag['intra_cos_rna']:.3f}/{diag['intra_cos_prot']:.3f} | "
          f"std/dim ARN/prot = {diag['std_dim_rna']:.3f}/{diag['std_dim_prot']:.3f}")
    report["diagnostics"] = diag

    # ---- labels (optionnel) ----
    labels_te = None
    if a.labels:
        df = pd.read_csv(a.labels)
        idc = "cell_id" if "cell_id" in df.columns else df.columns[0]
        lc = a.label_col if a.label_col in df.columns else df.columns[1]
        mp = dict(zip(df[idc].astype(str), df[lc].astype(str)))
        lab_all = np.array([mp.get(c, "NA") for c in cell_id], dtype=object)
        has = lab_all != "NA"
        k = a.n_clusters or len(set(lab_all[has].tolist()))
        print(f"\n=== 3. Clustering & sonde linéaire (labels: {has.sum()} cellules, {k} classes) ===")

        spaces = {
            "CLIP joint": (l2(Zr_tr + Zp_tr), l2(Zr_te + Zp_te)),
            "CLIP ARN": (Zr_tr, Zr_te),
            "CLIP protéine": (Zp_tr, Zp_te),
            "NOVAE brut (ARN)": (rna[tr], rna[te]),
            "protéine brute": (prot[tr], prot[te]),
        }
        tr_has, te_has = has[tr], has[te]
        y_tr, y_te = lab_all[tr][tr_has], lab_all[te][te_has]
        labels_te = lab_all[te]
        clu, prb = {}, {}
        print(f"  {'espace':18s}{'ARI':>8}{'NMI':>8}{'probe acc':>11}{'probe F1':>10}")
        for nm, (Ztr_s, Zte_s) in spaces.items():
            ari, nmi = cluster_scores(Zte_s[te_has], y_te, k)
            acc, f1 = linear_probe(Ztr_s[tr_has], y_tr, Zte_s[te_has], y_te)
            clu[nm] = {"ARI": ari, "NMI": nmi}; prb[nm] = {"acc": acc, "f1_macro": f1}
            print(f"  {nm:18s}{ari:8.3f}{nmi:8.3f}{acc:11.3f}{f1:10.3f}")
        report["clustering"] = clu; report["linear_probe"] = prb
    else:
        print("\n[info] pas de --labels : clustering/ARI/NMI et sonde linéaire ignorés "
              "(à relancer une fois CosMx annoté, ex. sc-concept).")

    # ---- 4. UMAP ----
    print("\n=== 4. UMAP / PCA (TEST) ===")
    plot_umap(Zr_te, Zp_te, out / "umap_test.png",
              labels=labels_te if (labels_te is not None) else None)

    json.dump(report, open(out / "eval_report.json", "w"), indent=2)
    print(f"\nRapport écrit : {out/'eval_report.json'}")
    # Verdict retrieval rapide vs CCA
    clip_r5 = 0.5 * (store["CLIP"]["ARN→prot"][5] + store["CLIP"]["prot→ARN"][5])
    cca_r5 = 0.5 * (store["CCA (linéaire)"]["ARN→prot"][5] + store["CCA (linéaire)"]["prot→ARN"][5])
    verdict = "✓ le CLIP bat la CCA" if clip_r5 > cca_r5 else "✗ le CLIP ne bat pas la CCA"
    print(f"R@5 test (moy.) : CLIP={100*clip_r5:.2f}%  vs  CCA={100*cca_r5:.2f}%  → {verdict}")


if __name__ == "__main__":
    main()
