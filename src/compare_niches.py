"""
compare_niches.py — Comparer des cartes de niches issues de RUNS différents.

Cas d'usage typique : comparer les niches multi-omiques (`clip_joint`) du CLIP **NOVAE**
et du CLIP **scConcept**, qui vivent dans deux runs distincts (checkpoints + paired-dirs
différents). L'ordre des `cell_id` étant identique entre les paired-dirs NOVAE et scConcept,
les `domains_*.npy` sauvés par eval_niches se superposent cellule à cellule.

On évalue toutes les cartes sur le MÊME graphe spatial (comparaison équitable) : FIDE,
entropie normalisée, heuristique, et ARI/NMI vs types cellulaires (optionnel), plus une
figure côte-à-côte.

Pré-requis : chaque `domains_*.npy` doit être aligné sur l'ordre des cellules de
`--paired-dir` (cas par défaut quand eval_niches a tourné avec `--cells all` sans
`scconcept_raw`, donc une carte de longueur N complète).

Exemple :
  # 1) générer les deux cartes clip_joint (sur ta machine, torch requis)
  python src/eval_niches.py --paired-dir data/processed/cosmx_breast \
    --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
    --ckpt runs/clip_cosmx_seed42/best.pt --spaces clip_joint --n-domains 10 \
    --outdir eval/niches_cosmx_novae
  python src/eval_niches.py --paired-dir data/processed/cosmx_breast_scconcept \
    --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
    --ckpt runs/clip_cosmx_scconcept_seed42/best.pt --spaces clip_joint --n-domains 10 \
    --outdir eval/niches_cosmx_scconcept

  # 2) comparer
  python src/compare_niches.py \
    --paired-dir data/processed/cosmx_breast \
    --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
    --labels-h5ad annotation_out/cosmx_annotated.h5ad \
    --maps NOVAE=eval/niches_cosmx_novae/domains_clip_joint.npy \
           scConcept=eval/niches_cosmx_scconcept/domains_clip_joint.npy \
    --outdir eval/compare_cosmx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Tout est importé d'eval_niches (aucune dépendance torch au niveau module).
from eval_niches import (
    ari_nmi,
    fide_score,
    heuristic_score,
    knn_graph,
    load_obs_aligned,
    load_obsm_aligned,
    load_views,
    normalized_entropy,
    plot_compare,
)


def parse_maps(items):
    """['NOVAE=path.npy', ...] -> dict {nom: chemin}."""
    out = {}
    for it in items:
        assert "=" in it, f"--maps attend 'nom=chemin.npy', reçu : {it!r}"
        name, path = it.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def main():
    p = argparse.ArgumentParser(description="Comparer des cartes de niches inter-runs.")
    p.add_argument("--paired-dir", required=True, help="référence pour l'ordre des cell_id + coords")
    p.add_argument("--spatial-h5ad", required=True)
    p.add_argument("--spatial-obsm", default="spatial_um")
    p.add_argument("--maps", nargs="+", required=True, help="nom=chemin.npy (2+ cartes)")
    p.add_argument("--labels-h5ad", default=None)
    p.add_argument("--label-col", default="cell_type_final")
    p.add_argument("--label-drop", default="incertaine")
    p.add_argument("--knn-graph", type=int, default=6)
    p.add_argument("--outdir", default="eval/niches_compare")
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rna, _, cell_id, split = load_views(a.paired_dir)
    N = len(cell_id)

    try:
        coords_all, found_sp = load_obsm_aligned(a.spatial_h5ad, a.spatial_obsm, cell_id, ndim=2)
    except KeyError:
        coords_all, found_sp = load_obsm_aligned(a.spatial_h5ad, "spatial", cell_id, ndim=2)
    labels_all = load_obs_aligned(a.labels_h5ad, a.label_col, cell_id) if a.labels_h5ad else None
    label_drop = {s.strip() for s in a.label_drop.split(",")} | {"NA"}

    maps = parse_maps(a.maps)
    domains_by_space = {}
    for name, path in maps.items():
        d = np.load(path)
        assert len(d) == N, (
            f"La carte '{name}' a {len(d)} cellules mais le paired-dir en a {N}. "
            f"Relance eval_niches avec --cells all et sans scconcept_raw pour une carte complète."
        )
        domains_by_space[name] = d.astype(np.int64)

    idx = np.where(found_sp)[0]
    coords = coords_all[idx]
    labels = labels_all[idx] if labels_all is not None else None
    print(f"cellules comparées : {len(idx)}/{N} | cartes : {list(domains_by_space)}")

    print(f"graphe spatial KNN (k={a.knn_graph}) ...")
    adj = knn_graph(coords, k=a.knn_graph)

    rows = []
    plot_maps = {}
    for name, d_full in domains_by_space.items():
        d = d_full[idx]
        ncl = len(np.unique(d))
        fide = fide_score(d, adj, n_classes=ncl)
        hent = normalized_entropy(d, ncl)
        heur = fide * hent
        ari, nmi = ari_nmi(d, labels, label_drop)
        rows.append({"carte": name, "n_dom": ncl, "FIDE": round(fide, 4),
                     "entropie_norm": round(hent, 4), "heuristique": round(heur, 4),
                     "ARI_types": (round(ari, 4) if ari is not None else None),
                     "NMI_types": (round(nmi, 4) if nmi is not None else None)})
        plot_maps[name] = d
        print(f"  {name:12s} n_dom={ncl} FIDE={fide:.4f} heuristique={heur:.4f}"
              + (f" ARI/NMI={ari:.3f}/{nmi:.3f}" if ari is not None else ""))

    plot_compare(coords, plot_maps, out / "niches_compare.png")

    table = pd.DataFrame(rows)
    print("\n=== COMPARAISON INTER-RUNS ===")
    print(table.to_string(index=False))
    best = table.loc[table["heuristique"].idxmax(), "carte"]
    print(f"\nMeilleure heuristique : {best}")
    table.to_csv(out / "compare_summary.csv", index=False)
    json.dump(rows, open(out / "compare_report.json", "w"), indent=2, default=str)
    print(f"Rapport : {out / 'compare_report.json'}")


if __name__ == "__main__":
    main()
