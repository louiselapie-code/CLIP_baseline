"""
eval_niches.py — Évaluation des NICHES multi-omiques via la « fin de NOVAE ».

But (cf. la question : NOVAE améliore-t-il les niches multi-omiques ?) :
on prend des embeddings cellulaires, on leur applique la tête de niches NOVAE
(niches.py), puis on évalue la qualité spatiale des niches obtenues — sans label,
avec les métriques de NOVAE (FIDE, entropie normalisée, heuristique ; JSD si plusieurs
slides).

On compare plusieurs ESPACES d'embeddings, tous évalués sur le MÊME graphe spatial
(comparaison équitable) :
    novae_raw     : paired_rna.npy (NOVAE gelé, 64d) — ARN seul, référence « NOVAE seul »
    clip_rna      : z_r = rna_head(NOVAE)        (256d, sortie CLIP côté ARN)
    clip_prot     : z_p = protein_tower(prot)    (256d, sortie CLIP côté protéine)
    clip_joint    : l2(z_r + z_p)                (256d, MULTI-OMIQUE) — l'espace clé
    scconcept_raw : obsm X_scConcept d'un .h5ad  (512d, encodeur ARN alternatif)

Les coordonnées spatiales (obsm['spatial_um'], en µm) sont alignées par cell_id depuis
le .h5ad « *_rna_with_spatial_split_seed42.h5ad ».

Exemples :
  # NOVAE seul vs CLIP multi-omique (CosMx)
  python eval_niches.py \
    --paired-dir data/processed/cosmx_breast \
    --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
    --ckpt results/runs/clip_cosmx_seed42/best.pt \
    --spaces novae_raw,clip_rna,clip_prot,clip_joint \
    --n-domains 10 --outdir results/eval/niches_cosmx

  # Ajouter scConcept brut comme comparaison d'encodeur ARN
  python eval_niches.py --paired-dir data/processed/xenium_renal \
    --spatial-h5ad data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
    --ckpt results/runs/clip_xenium_seed42/best.pt \
    --scconcept-h5ad annotation/xenium_annotated.h5ad \
    --spaces novae_raw,clip_joint,scconcept_raw --n-domains 10 --outdir results/eval/niches_xenium
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import niches

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Chargement (léger : pas de torch tant qu'on ne demande pas d'espace CLIP)
# --------------------------------------------------------------------------- #
def load_views(paired_dir):
    d = Path(paired_dir)
    rna = np.load(d / "paired_rna.npy").astype(np.float32)
    prot = np.load(d / "paired_protein.npy").astype(np.float32)
    cells = pd.read_csv(d / "paired_cells.csv")
    return rna, prot, cells["cell_id"].astype(str).to_numpy(), cells["split"].to_numpy()


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_obsm_aligned(h5ad, key, cell_id, ndim=None):
    """Charge obsm[key] d'un .h5ad et l'aligne sur l'ordre des cellules appariées (par cell_id).

    Calqué sur eval_intra_type.load_obsm_aligned. Retourne (array (N, d), mask trouvé).
    """
    import anndata as ad

    a = ad.read_h5ad(h5ad, backed="r")
    if key not in a.obsm:
        raise KeyError(f"obsm['{key}'] absent de {h5ad}. Disponibles : {list(a.obsm)}")
    emb = np.asarray(a.obsm[key])
    if ndim is not None:
        emb = emb[:, :ndim]
    ids = a.obs["cell_id"].astype(str) if "cell_id" in a.obs.columns else a.obs.index.astype(str)
    idx = {str(c): i for i, c in enumerate(ids)}
    rows = np.array([idx.get(str(c), -1) for c in cell_id])
    found = rows >= 0
    out = np.zeros((len(cell_id), emb.shape[1]), dtype=np.float32)
    out[found] = emb[rows[found]].astype(np.float32)
    print(f"  [{key}] {found.sum()}/{len(cell_id)} cellules trouvées ({found.mean() * 100:.1f}%)")
    return out, found


def load_obs_aligned(h5ad, col, cell_id):
    """Charge obs[col] aligné par cell_id (ex. slide). Retourne array objet (N,) ou None si absent."""
    import anndata as ad

    a = ad.read_h5ad(h5ad, backed="r")
    if col not in a.obs.columns:
        return None
    ids = a.obs["cell_id"].astype(str) if "cell_id" in a.obs.columns else a.obs.index.astype(str)
    mp = dict(zip(ids.astype(str), a.obs[col].astype(str)))
    return np.array([mp.get(str(c), "NA") for c in cell_id], dtype=object)


# --------------------------------------------------------------------------- #
# Graphe spatial (partagé entre les espaces → comparaison équitable)
# --------------------------------------------------------------------------- #
def knn_graph(coords, k=6):
    """Graphe de voisinage spatial KNN symétrisé (connectivité), sparse CSR booléen."""
    from sklearn.neighbors import kneighbors_graph

    g = kneighbors_graph(np.asarray(coords, dtype=np.float64), n_neighbors=k, mode="connectivity")
    g = g.maximum(g.T)  # symétrise (arêtes non orientées)
    g.setdiag(0)
    g.eliminate_zeros()
    return g.tocsr()


# --------------------------------------------------------------------------- #
# Métriques NOVAE (port de novae/monitor/eval.py)
# --------------------------------------------------------------------------- #
def fide_score(domains, adj, n_classes=None):
    """F1 des arêtes intra-domaine (FIDE). Élevé = domaines spatialement continus."""
    from sklearn.metrics import f1_score

    il, ir = adj.nonzero()
    cl, cr = np.asarray(domains)[il], np.asarray(domains)[ir]
    f1 = f1_score(cl, cr, average=None)
    if n_classes is None:
        return float(np.mean(f1))
    return float(np.pad(f1, (0, max(0, n_classes - len(f1)))).mean())


def shannon_entropy(distribution):
    distribution = np.asarray(distribution, dtype=np.float64)
    return float(-(distribution * np.log2(distribution + EPS)).sum())


def normalized_entropy(domains, n_classes):
    counts = np.bincount(np.asarray(domains), minlength=n_classes).astype(np.float64)
    dist = counts / counts.sum()
    return float(shannon_entropy(dist) / np.log2(n_classes))


def heuristic_score(domains, adj, n_classes):
    """fide * entropie_normalisée : compromis continuité spatiale / diversité (NOVAE)."""
    return float(fide_score(domains, adj, n_classes=n_classes) * normalized_entropy(domains, n_classes))


def jensen_shannon_divergence(domains, slide_labels):
    """JSD des distributions de domaines entre slides. Bas = bon mélange inter-slides.

    Nécessite >= 2 slides. Port de novae/monitor/eval.py.
    """
    slides = [s for s in pd.unique(slide_labels) if s != "NA"]
    if len(slides) < 2:
        return None
    cats = sorted(np.unique(domains).tolist())
    dists = []
    for s in slides:
        m = slide_labels == s
        counts = np.array([(domains[m] == c).sum() for c in cats], dtype=np.float64)
        dists.append(counts)
    dists = np.array(dists)
    dists = dists / dists.sum(1, keepdims=True)
    mean = dists.mean(0)
    return float(shannon_entropy(mean) - np.mean([shannon_entropy(d) for d in dists]))


# --------------------------------------------------------------------------- #
# Construction des espaces d'embeddings
# --------------------------------------------------------------------------- #
def build_space(name, rna, prot, cell_id, args):
    """Retourne (embeddings (N, D), mask_valide (N,))."""
    N = len(rna)
    if name == "novae_raw":
        return rna, np.ones(N, bool)
    if name == "scconcept_raw":
        assert args.scconcept_h5ad, "--scconcept-h5ad requis pour l'espace scconcept_raw"
        emb, found = load_obsm_aligned(args.scconcept_h5ad, args.scconcept_obsm, cell_id)
        return emb, found
    if name in ("clip_rna", "clip_prot", "clip_joint"):
        assert args.ckpt, f"--ckpt requis pour l'espace {name}"
        # Import paresseux : torch n'est nécessaire que pour les espaces CLIP.
        from evaluate import load_model, embed  # noqa: E402
        import torch

        device = torch.device(args.device)
        model = load_model(args.ckpt, device)
        zr, zp = embed(model, rna, prot, device)
        if name == "clip_rna":
            return zr, np.ones(N, bool)
        if name == "clip_prot":
            return zp, np.ones(N, bool)
        return l2(zr + zp), np.ones(N, bool)  # clip_joint
    raise ValueError(f"espace inconnu : {name}")


# --------------------------------------------------------------------------- #
# Visualisation spatiale des niches
# --------------------------------------------------------------------------- #
def plot_niches(coords, domains, outpath, title, max_points=60000, seed=0):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  [info] matplotlib indisponible, plot ignoré ({e}).")
        return
    rng = np.random.default_rng(seed)
    idx = np.arange(len(coords))
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[idx, 0], coords[idx, 1], c=domains[idx], cmap="tab20", s=2, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, label="niche")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  carte spatiale : {outpath}")


def plot_compare(coords, domains_by_space, outpath, max_points=60000, seed=0):
    """Cartes spatiales des niches côte à côte (un panneau par espace), mêmes coords."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  [info] matplotlib indisponible, figure comparative ignorée ({e}).")
        return
    names = list(domains_by_space)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(coords))
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 6), squeeze=False)
    for ax, name in zip(axes[0], names):
        d = np.asarray(domains_by_space[name])
        ax.scatter(coords[idx, 0], coords[idx, 1], c=d[idx], cmap="tab20", s=2, alpha=0.7)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  figure comparative : {outpath}")


def ari_nmi(domains, labels, drop):
    """ARI/NMI entre niches et labels de types cellulaires, hors labels exclus. (None si indispo.)"""
    if labels is None:
        return None, None
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    keep = np.array([str(l) not in drop for l in labels])
    if keep.sum() == 0 or len(np.unique(labels[keep])) < 2:
        return None, None
    return (float(adjusted_rand_score(labels[keep], np.asarray(domains)[keep])),
            float(normalized_mutual_info_score(labels[keep], np.asarray(domains)[keep])))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Évaluation des niches multi-omiques (fin de NOVAE).")
    p.add_argument("--paired-dir", required=True)
    p.add_argument("--spatial-h5ad", required=True, help=".h5ad avec obs.cell_id + obsm spatial (µm)")
    p.add_argument("--spatial-obsm", default="spatial_um", help="clé obsm des coords (fallback: spatial)")
    p.add_argument("--ckpt", default=None, help="checkpoint CLIP (requis pour les espaces clip_*)")
    p.add_argument("--spaces", default="novae_raw,clip_joint",
                   help="liste séparée par des virgules : novae_raw,clip_rna,clip_prot,clip_joint,scconcept_raw")
    p.add_argument("--scconcept-h5ad", default=None)
    p.add_argument("--scconcept-obsm", default="X_scConcept")
    p.add_argument("--num-prototypes", type=int, default=512)
    p.add_argument("--n-domains", type=int, default=10)
    p.add_argument("--niche-method", default="hierarchical", choices=["hierarchical", "leiden"])
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--assign", default="argmax", choices=["argmax", "sinkhorn"])
    p.add_argument("--smooth-knn", type=int, default=0, help="lissage spatial sur k voisins (0 = off)")
    p.add_argument("--knn-graph", type=int, default=6, help="k du graphe spatial pour le FIDE")
    p.add_argument("--cells", default="all", choices=["all", "train", "test"])
    p.add_argument("--slide-key", default=None, help="colonne obs pour le JSD inter-slides (optionnel)")
    p.add_argument("--labels-h5ad", default=None, help=".h5ad avec obs[label-col] -> ARI/NMI niches vs types cellulaires")
    p.add_argument("--label-col", default="cell_type_final")
    p.add_argument("--label-drop", default="incertaine", help="labels exclus de l'ARI/NMI (virgules), ex. incertaine,NA")
    p.add_argument("--max-fit-cells", type=int, default=100_000, help="sous-échantillon KMeans")
    p.add_argument("--outdir", default="results/eval/niches")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rna, prot, cell_id, split = load_views(a.paired_dir)
    N = len(rna)
    print(f"N={N} | rna_dim={rna.shape[1]} prot_dim={prot.shape[1]} | cells={a.cells}")

    # ---- coordonnées spatiales ----
    try:
        coords_all, found_sp = load_obsm_aligned(a.spatial_h5ad, a.spatial_obsm, cell_id, ndim=2)
    except KeyError:
        print(f"  [info] obsm['{a.spatial_obsm}'] absent, fallback sur 'spatial'.")
        coords_all, found_sp = load_obsm_aligned(a.spatial_h5ad, "spatial", cell_id, ndim=2)
    slide_all = load_obs_aligned(a.spatial_h5ad, a.slide_key, cell_id) if a.slide_key else None
    labels_all = load_obs_aligned(a.labels_h5ad, a.label_col, cell_id) if a.labels_h5ad else None
    label_drop = {s.strip() for s in a.label_drop.split(",")} | {"NA"}

    # ---- masque cellules ----
    base_mask = found_sp.copy()
    if a.cells == "train":
        base_mask &= split == "train"
    elif a.cells == "test":
        base_mask &= split == "test"

    spaces = [s.strip() for s in a.spaces.split(",") if s.strip()]

    # Si scconcept demandé, on restreint le masque GLOBAL aux cellules trouvées (mêmes
    # cellules pour tous les espaces → comparaison équitable sur le même graphe).
    if "scconcept_raw" in spaces:
        _, found_sc = load_obsm_aligned(a.scconcept_h5ad, a.scconcept_obsm, cell_id)
        base_mask &= found_sc

    idx = np.where(base_mask)[0]
    coords = coords_all[idx]
    slide = slide_all[idx] if slide_all is not None else None
    labels = labels_all[idx] if labels_all is not None else None
    print(f"  cellules évaluées : {len(idx)}")

    # ---- graphe spatial partagé ----
    print(f"  graphe spatial KNN (k={a.knn_graph}) ...")
    adj = knn_graph(coords, k=a.knn_graph)

    # ---- boucle sur les espaces ----
    report = {"config": vars(a), "n_cells_eval": int(len(idx)), "spaces": {}}
    rows = []
    domains_by_space = {}
    for name in spaces:
        print(f"\n=== espace : {name} ===")
        emb_full, valid = build_space(name, rna, prot, cell_id, a)
        E = emb_full[idx]
        assert valid[idx].all(), f"cellules invalides dans l'espace {name} (incohérence de masque)"

        if a.smooth_knn > 0:
            print(f"  lissage spatial KNN (k={a.smooth_knn}) ...")
            E = niches.spatial_smooth(E, coords, k=a.smooth_knn)

        res = niches.assign_niches(
            E,
            num_prototypes=a.num_prototypes,
            n_domains=a.n_domains,
            niche_method=a.niche_method,
            resolution=a.resolution,
            assign=a.assign,
            sample_cells=a.max_fit_cells,
            seed=a.seed,
        )
        dom = res.domains
        ncl = res.n_domains

        fide = fide_score(dom, adj, n_classes=ncl)
        hent = normalized_entropy(dom, ncl)
        heur = fide * hent
        jsd = jensen_shannon_divergence(dom, slide) if slide is not None else None
        ari, nmi = ari_nmi(dom, labels, label_drop)

        sizes = np.bincount(dom).tolist()
        report["spaces"][name] = {
            "dim": int(E.shape[1]), "n_domains": int(ncl), "level": int(res.level),
            "n_prototypes_used": int(len(np.unique(res.leaves))),
            "FIDE": fide, "entropie_norm": hent, "heuristique": heur, "JSD": jsd,
            "ARI_vs_types": ari, "NMI_vs_types": nmi,
            "tailles_domaines": sizes,
        }
        rows.append({"espace": name, "dim": E.shape[1], "n_dom": ncl,
                     "FIDE": round(fide, 4), "entropie_norm": round(hent, 4),
                     "heuristique": round(heur, 4),
                     "JSD": (round(jsd, 4) if jsd is not None else None),
                     "ARI_types": (round(ari, 4) if ari is not None else None),
                     "NMI_types": (round(nmi, 4) if nmi is not None else None)})
        print(f"  n_domaines={ncl} (niveau {res.level}) | FIDE={fide:.4f} "
              f"entropie_norm={hent:.4f} heuristique={heur:.4f}"
              + (f" JSD={jsd:.4f}" if jsd is not None else "")
              + (f" | ARI/NMI vs types={ari:.3f}/{nmi:.3f}" if ari is not None else ""))

        if not a.no_plot:
            plot_niches(coords, dom, out / f"niches_{name}.png",
                        title=f"{name} — {ncl} niches (FIDE={fide:.3f})")
        # sauvegarde des assignations par cellule
        np.save(out / f"domains_{name}.npy", dom)
        domains_by_space[name] = dom

    # ---- figure comparative côte-à-côte (tous les espaces du run) ----
    if not a.no_plot and len(domains_by_space) >= 2:
        plot_compare(coords, domains_by_space, out / "niches_compare.png")

    # ---- table récap + verdict ----
    table = pd.DataFrame(rows)
    print("\n=== RÉCAPITULATIF NICHES ===")
    print(table.to_string(index=False))
    best_heur = table.loc[table["heuristique"].idxmax(), "espace"]
    best_fide = table.loc[table["FIDE"].idxmax(), "espace"]
    print(f"\nMeilleure heuristique : {best_heur} | meilleur FIDE : {best_fide}")
    print("Lecture : FIDE haut = niches spatialement continues ; entropie_norm haute = niches")
    print("équilibrées ; heuristique = compromis des deux. Compare 'novae_raw' (ARN seul) à")
    print("'clip_joint' (multi-omique) pour voir si le CLIP améliore les niches.")
    if labels is not None:
        print("ARI/NMI vs types : un score BAS est plutôt sain (niches = voisinages, pas types")
        print("cellulaires). Un score élevé = l'espace ne fait que du typage cellulaire, pas des niches.")

    table.to_csv(out / "niches_summary.csv", index=False)
    json.dump(report, open(out / "niches_report.json", "w"), indent=2, default=str)
    print(f"\nRapport : {out / 'niches_report.json'}")


if __name__ == "__main__":
    main()
