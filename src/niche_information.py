"""
niche_information.py — Les niches MULTI-OMIQUES portent-elles plus d'information
                       que les niches UNI-OMIQUES ?

FIDE ne répond PAS à cette question (il ne mesure que la continuité spatiale). Ici on
mesure l'INFORMATION, avec une idée simple et défendable :

    une niche est "informative sur une modalité" si elle prédit bien cette modalité,
    c.-à-d. si les cellules d'une même niche se ressemblent dans cette modalité.

On quantifie ça par la **variance expliquée** (η², multivarié) de chaque modalité BRUTE
(protéine ; ARN = comptages) par une partition en niches. Puis on compare 3 partitions
(même nb de domaines, mêmes cellules) :

    niche_ARN  (uni-omique, depuis paired_rna = NOVAE)
    niche_prot (uni-omique, depuis paired_protein = intensités brutes)
    niche_joint(multi-omique, depuis ton clip_joint : --joint-domains *.npy)

Logique du test :
  - une niche uni-omique est bonne sur SA modalité, mauvaise sur l'autre ;
  - si la niche JOINTE est bonne sur les DEUX à la fois (coin haut-droit du "plan
    d'information"), alors elle intègre réellement l'info des deux → plus informative.

Verdict (non circulaire pour la partie clé) :
  EV(joint, protéine) > EV(niche_ARN, protéine)   → la jointe capte du signal protéique
                                                     que la niche ARN n'a pas, ET
  EV(joint, ARN)      > EV(niche_prot, ARN)        → ...du signal ARN que la niche prot n'a pas.
  Si les deux tiennent : le multi-omique > chaque uni-omique en information.

Compléments : raffinement croisé (entropies conditionnelles, AMI, contingence) et un
zoom sur UNE niche ARN scindée par la jointe, avec test de permutation montrant que les
sous-niches diffèrent réellement en protéine.

Sans torch : la partition jointe est lue depuis le .npy sauvé par eval_niches ; les
partitions ARN/protéine sont (re)calculées depuis les vues appariées.

Exemple :
  python src/niche_information.py \
    --paired-dir data/processed/cosmx_breast \
    --rna-counts-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
    --joint-domains eval/niches_cosmx/domains_clip_joint.npy \
    --n-domains 10 --outdir eval/info_cosmx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import niches
from eval_niches import load_views, load_obsm_aligned

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Cibles "modalité brute"
# --------------------------------------------------------------------------- #
def standardize(M):
    """z-score par colonne, retire les colonnes de variance nulle."""
    M = np.asarray(M, dtype=np.float64)
    sd = M.std(0)
    M = M[:, sd > EPS]
    return (M - M.mean(0)) / (M.std(0) + EPS)


def rna_counts_target(h5ad, cell_id, n_pcs=50, fit_cells=40000, batch=4096, seed=0):
    """Cible ARN INDÉPENDANTE : comptages bruts -> log1p -> PCA incrémentale (n_pcs).

    Lecture frugale par morceaux via anndata en mode `backed` (tranches paresseuses) : gère
    X stocké DENSE (ex. CosMx, 152k x 20k = 11.6 Go) comme SPARSE/CSR (ex. Xenium). float32,
    petits morceaux, ajustement sur un sous-échantillon puis transformation de tout. Aligné
    ensuite par cell_id. Sur une machine à RAM limitée, baisse `batch` ou `--svd-pcs`.
    """
    import anndata as ad
    import scipy.sparse as sp
    from sklearn.decomposition import IncrementalPCA

    a = ad.read_h5ad(h5ad, backed="r")
    ids = a.obs["cell_id"].astype(str) if "cell_id" in a.obs.columns else a.obs.index.astype(str)
    n_obs, n_vars = a.n_obs, a.n_vars
    n_pcs = min(n_pcs, n_vars - 1, batch - 1)

    def read_log1p(s, e):
        chunk = a.X[s:e]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
        return np.log1p(np.asarray(chunk, dtype=np.float32))

    ipca = IncrementalPCA(n_components=n_pcs)
    n_batches = int(np.ceil(n_obs / batch))
    stride = max(1, int(np.ceil(n_obs / max(fit_cells, batch))))
    for bi in range(0, n_batches, stride):
        s = bi * batch
        chunk = read_log1p(s, min(s + batch, n_obs))
        if len(chunk) >= n_pcs:
            ipca.partial_fit(chunk)
        del chunk
    Z = np.empty((n_obs, n_pcs), dtype=np.float32)
    for s in range(0, n_obs, batch):
        e = min(s + batch, n_obs)
        Z[s:e] = ipca.transform(read_log1p(s, e)).astype(np.float32)

    pos = {str(c): i for i, c in enumerate(ids)}
    rows = np.array([pos.get(str(c), -1) for c in cell_id])
    found = rows >= 0
    out = np.zeros((len(cell_id), n_pcs), dtype=np.float64)
    out[found] = Z[rows[found]]
    print(f"  [ARN counts] {found.sum()}/{len(cell_id)} cellules | PCA {n_pcs} comp.")
    return standardize(out), found


# --------------------------------------------------------------------------- #
# Variance expliquée (η² multivarié) = information d'une partition sur une modalité
# --------------------------------------------------------------------------- #
def explained_variance(labels, M):
    """Fraction de la variance totale de M expliquée par la partition `labels` (0..1).

    EV = 1 - SS_within / SS_total. EV élevé = cellules d'une même niche proches dans M.
    """
    labels = np.asarray(labels)
    M = np.asarray(M, dtype=np.float64)
    gm = M.mean(0)
    ss_total = ((M - gm) ** 2).sum()
    ss_within = 0.0
    for g in np.unique(labels):
        Mi = M[labels == g]
        if len(Mi):
            ss_within += ((Mi - Mi.mean(0)) ** 2).sum()
    return float(1.0 - ss_within / (ss_total + EPS))


def random_partition_floor(n, k, M, seed=0, reps=3):
    """EV moyen d'une partition aléatoire à k classes (plancher de référence)."""
    rng = np.random.default_rng(seed)
    return float(np.mean([explained_variance(rng.integers(0, k, n), M) for _ in range(reps)]))


# --------------------------------------------------------------------------- #
# Raffinement croisé entre partitions
# --------------------------------------------------------------------------- #
def partition_relations(a, b):
    """Entropies conditionnelles H(a|b), H(b|a) (bits) et AMI entre deux partitions."""
    from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix

    C = contingency_matrix(a, b).astype(np.float64)
    N = C.sum()
    pa = C.sum(1) / N
    pb = C.sum(0) / N
    Ha = -(pa * np.log2(pa + EPS)).sum()
    Hb = -(pb * np.log2(pb + EPS)).sum()
    mi_nats = mutual_info_score(a, b)
    mi_bits = mi_nats / np.log(2)
    return {
        "H_a": float(Ha), "H_b": float(Hb),
        "H_a_given_b": float(Ha - mi_bits),  # incertitude restante sur a si on connaît b
        "H_b_given_a": float(Hb - mi_bits),
        "MI_bits": float(mi_bits),
        "AMI": float(adjusted_mutual_info_score(a, b)),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_information_plane(ev, outpath):
    """Plan d'information : EV(ARN) en x, EV(protéine) en y, un point par partition."""
    try:
        plt = _mpl()
    except Exception as e:  # pragma: no cover
        print(f"  [info] matplotlib indisponible ({e})."); return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = {"niche_ARN": "tab:blue", "niche_prot": "tab:orange",
              "niche_joint": "tab:green", "aléatoire": "grey"}
    for name, (x, y) in ev.items():
        ax.scatter(x, y, s=120, color=colors.get(name, "k"), zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=11)
    ax.set_xlabel("EV(ARN)  — information sur l'ARN")
    ax.set_ylabel("EV(protéine)  — information sur la protéine")
    ax.set_title("Plan d'information des niches\n(haut-droit = informe sur les DEUX modalités)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)
    print(f"  plan d'information : {outpath}")


def plot_contingency(a, b, outpath, xlabel, ylabel):
    """Heatmap de contingence (lignes normalisées) : comment a se répartit dans b."""
    try:
        plt = _mpl()
    except Exception as e:  # pragma: no cover
        print(f"  [info] matplotlib indisponible ({e})."); return
    ct = pd.crosstab(pd.Series(a, name=ylabel), pd.Series(b, name=xlabel), normalize="index")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ct.values, aspect="auto", cmap="viridis")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"Répartition de chaque «{ylabel}» dans les «{xlabel}»\n(une ligne étalée = niche scindée)")
    fig.colorbar(im, ax=ax, label="fraction de la ligne")
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)
    print(f"  contingence : {outpath}")


def refinement_drilldown(rna_niche, joint_niche, protein, outpath, min_sub=200, seed=0):
    """Zoom : la niche ARN la plus scindée par la jointe ; les sous-niches diffèrent-elles en protéine ?

    Retourne un dict (niche choisie, p-value de permutation, variance inter-sous-niches).
    """
    prot = standardize(protein)
    # niche ARN la plus "scindée" : entropie de la distribution des sous-niches jointes
    best_r, best_H, best_sub = None, -1, None
    for r in np.unique(rna_niche):
        m = rna_niche == r
        sub = joint_niche[m]
        vals, cnts = np.unique(sub, return_counts=True)
        keep = vals[cnts >= min_sub]
        if len(keep) < 2:
            continue
        p = cnts[cnts >= min_sub] / cnts[cnts >= min_sub].sum()
        H = -(p * np.log2(p + EPS)).sum()
        if H > best_H:
            best_r, best_H, best_sub = r, H, keep
    if best_r is None:
        print("  [info] aucune niche ARN nettement scindée (min_sub trop grand ?).")
        return None

    m = (rna_niche == best_r) & np.isin(joint_niche, best_sub)
    sub = joint_niche[m]
    Pm = prot[m]
    # variance protéique INTER-sous-niches (ce que la jointe "ajoute" dans cette niche ARN)
    between = explained_variance(sub, Pm)
    rng = np.random.default_rng(seed)
    null = [explained_variance(rng.permutation(sub), Pm) for _ in range(200)]
    pval = (1 + np.sum(np.asarray(null) >= between)) / (1 + len(null))

    # figure : profil protéique moyen par sous-niche
    try:
        plt = _mpl()
        order = sorted(best_sub.tolist())
        means = np.vstack([Pm[sub == s].mean(0) for s in order])
        fig, ax = plt.subplots(figsize=(min(14, 0.22 * Pm.shape[1] + 3), 0.6 * len(order) + 2))
        im = ax.imshow(means, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([f"joint {s}" for s in order])
        ax.set_xlabel("canaux protéiques (z-score)")
        ax.set_title(f"Niche ARN {best_r} scindée par la jointe — profils protéiques moyens\n"
                     f"variance inter-sous-niches EV={between:.3f}, permutation p={pval:.3g}")
        fig.colorbar(im, ax=ax, label="z-score moyen")
        fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)
        print(f"  drill-down : {outpath}")
    except Exception as e:  # pragma: no cover
        print(f"  [info] figure drill-down ignorée ({e}).")

    return {"niche_ARN_choisie": int(best_r), "n_sous_niches": int(len(best_sub)),
            "n_cellules": int(m.sum()), "EV_protein_inter_sous_niches": float(between),
            "permutation_p": float(pval)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Information des niches multi- vs uni-omiques.")
    p.add_argument("--paired-dir", required=True, help="rna=NOVAE (paired_rna), protein=intensités")
    p.add_argument("--joint-domains", required=True, help=".npy des niches multi-omiques (clip_joint)")
    p.add_argument("--rna-domains", default=None, help=".npy niche ARN (sinon recalculée depuis paired_rna)")
    p.add_argument("--prot-domains", default=None, help=".npy niche protéine (sinon recalculée depuis paired_protein)")
    p.add_argument("--rna-counts-h5ad", default=None, help="h5ad de comptages bruts (cible ARN indépendante)")
    p.add_argument("--rna-target", default="novae", choices=["counts", "novae"],
                   help="cible ARN : novae (paired_rna, rapide ; un peu circulaire) ou counts "
                        "(comptages bruts du h5ad, indépendant mais plus lourd)")
    p.add_argument("--num-prototypes", type=int, default=512)
    p.add_argument("--n-domains", type=int, default=10)
    p.add_argument("--svd-pcs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="eval/niche_info")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    rna, prot, cell_id, split = load_views(a.paired_dir)
    N = len(rna)
    print(f"N={N} | rna_dim={rna.shape[1]} prot_dim={prot.shape[1]}")

    # ---- partitions (mêmes n_domains / prototypes pour une comparaison équitable) ----
    def get_or_compute(path, source, tag):
        if path:
            d = np.load(path)
            assert len(d) == N, f"{tag}: {len(d)} != {N} cellules"
            return d.astype(np.int64)
        print(f"  calcul niche {tag} ...")
        return niches.assign_niches(source, num_prototypes=a.num_prototypes,
                                    n_domains=a.n_domains, seed=a.seed).domains

    niche_joint = get_or_compute(a.joint_domains, None, "joint")
    niche_rna = get_or_compute(a.rna_domains, rna, "ARN")
    niche_prot = get_or_compute(a.prot_domains, prot, "protéine")

    # ---- cibles "modalité brute" ----
    M_prot = standardize(prot)
    if a.rna_target == "counts":
        assert a.rna_counts_h5ad, "--rna-counts-h5ad requis pour la cible ARN 'counts'"
        M_rna, _ = rna_counts_target(a.rna_counts_h5ad, cell_id, n_pcs=a.svd_pcs)
        rna_target_name = "ARN counts (indépendant)"
    else:
        M_rna = standardize(rna)
        rna_target_name = "ARN NOVAE (paired_rna, circulaire)"

    # ---- variance expliquée : information de chaque partition sur chaque modalité ----
    parts = {"niche_ARN": niche_rna, "niche_prot": niche_prot, "niche_joint": niche_joint}
    ev_plane, rows = {}, []
    for name, lab in parts.items():
        ev_r = explained_variance(lab, M_rna)
        ev_p = explained_variance(lab, M_prot)
        ev_plane[name] = (ev_r, ev_p)
        rows.append({"partition": name, "EV_ARN": round(ev_r, 4), "EV_protéine": round(ev_p, 4)})
    fl_r = random_partition_floor(N, a.n_domains, M_rna, a.seed)
    fl_p = random_partition_floor(N, a.n_domains, M_prot, a.seed)
    ev_plane["aléatoire"] = (fl_r, fl_p)
    rows.append({"partition": "aléatoire", "EV_ARN": round(fl_r, 4), "EV_protéine": round(fl_p, 4)})

    table = pd.DataFrame(rows)
    print(f"\n=== VARIANCE EXPLIQUÉE (cible ARN = {rna_target_name}) ===")
    print(table.to_string(index=False))

    # ---- raffinement croisé ----
    rel_rj = partition_relations(niche_rna, niche_joint)
    rel_pj = partition_relations(niche_prot, niche_joint)
    print("\n=== RAFFINEMENT (entropies en bits) ===")
    print(f"  H(niche_ARN | joint)  = {rel_rj['H_a_given_b']:.3f}   (bas = la jointe contient la niche ARN)")
    print(f"  H(joint | niche_ARN)  = {rel_rj['H_b_given_a']:.3f}   (haut = la jointe AJOUTE des distinctions)")
    print(f"  H(niche_prot | joint) = {rel_pj['H_a_given_b']:.3f}   (bas = la jointe contient la niche protéine)")
    print(f"  H(joint | niche_prot) = {rel_pj['H_b_given_a']:.3f}")
    print(f"  AMI(ARN, joint)={rel_rj['AMI']:.3f} | AMI(prot, joint)={rel_pj['AMI']:.3f}")

    # ---- verdict ----
    add_prot = ev_plane["niche_joint"][1] - ev_plane["niche_ARN"][1]   # la jointe ajoute-t-elle de la protéine vs niche ARN ?
    add_rna = ev_plane["niche_joint"][0] - ev_plane["niche_prot"][0]   # ...et de l'ARN vs niche protéine ?
    verdict = (add_prot > 0) and (add_rna > 0)
    print("\n=== VERDICT ===")
    print(f"  EV(joint, protéine) - EV(niche_ARN, protéine) = {add_prot:+.4f}  "
          f"({'la jointe capte du signal protéique que la niche ARN rate' if add_prot>0 else 'pas de gain protéique'})")
    print(f"  EV(joint, ARN)      - EV(niche_prot, ARN)     = {add_rna:+.4f}  "
          f"({'la jointe capte du signal ARN que la niche protéine rate' if add_rna>0 else 'pas de gain ARN'})")
    print("  -> " + ("✓ Les niches MULTI-OMIQUES portent plus d'information que chaque uni-omique."
                     if verdict else
                     "✗ Pas de preuve nette d'un gain d'information multi-omique (voir nuances)."))

    # ---- figures ----
    drill = None
    if not a.no_plot:
        plot_information_plane(ev_plane, out / "information_plane.png")
        plot_contingency(niche_rna, niche_joint, out / "contingency_rna_joint.png",
                         xlabel="niche_joint", ylabel="niche_ARN")
        plot_contingency(niche_prot, niche_joint, out / "contingency_prot_joint.png",
                         xlabel="niche_joint", ylabel="niche_prot")
        drill = refinement_drilldown(niche_rna, niche_joint, prot, out / "refinement_example.png", seed=a.seed)
        if drill:
            print(f"  drill-down : niche ARN {drill['niche_ARN_choisie']} scindée en "
                  f"{drill['n_sous_niches']} sous-niches, EV protéique inter={drill['EV_protein_inter_sous_niches']:.3f}, "
                  f"p={drill['permutation_p']:.3g}")

    # ---- sauvegardes ----
    report = {"rna_target": rna_target_name, "n_cells": int(N), "n_domains": a.n_domains,
              "explained_variance": {k: {"EV_ARN": v[0], "EV_protein": v[1]} for k, v in ev_plane.items()},
              "verdict_multiomique_plus_informatif": bool(verdict),
              "gain_protein_vs_niche_ARN": float(add_prot), "gain_ARN_vs_niche_prot": float(add_rna),
              "relations": {"ARN_vs_joint": rel_rj, "prot_vs_joint": rel_pj},
              "drilldown": drill}
    table.to_csv(out / "information_summary.csv", index=False)
    json.dump(report, open(out / "information_report.json", "w"), indent=2, default=str)
    print(f"\nRapport : {out / 'information_report.json'}")


if __name__ == "__main__":
    main()
