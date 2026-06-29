"""
pathology_validation.py — Valider les niches contre des annotations de PATHOLOGIE (GeoJSON).

Vérité terrain externe : un·e pathologiste a tracé des régions (tumeur, stroma, immunitaire…)
sur le H&E. On assigne chaque cellule à sa région (point-dans-polygone), puis on mesure
l'accord entre chaque partition de niches et la pathologie (ARI/NMI/homogénéité). C'est LA
validation objective (≠ FIDE qui est interne).

Alignement : le GeoJSON est en pixels H&E ; la matrice affine 3×3 (imagealignment.csv) +
le facteur µm/pixel (Xenium = 0.2125) ramènent les polygones dans le repère µm des cellules.
Le bon sens de la transfo a été déterminé empiriquement (les régions tombent sur les bons
types cellulaires : Tumor→malignes, Immune→lymphocytes).

Exemple :
  python src/pathology_validation.py \
    --paired-dir data/processed/xenium_renal \
    --spatial-h5ad data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
    --geojson data/raw/xenium_renal/supplemental/Xenium_V1_Human_Kidney_FFPE_Protein_updated_annotation.geojson \
    --alignment data/raw/xenium_renal/supplemental/Xenium_V1_Human_Kidney_FFPE_Protein_updated_he_imagealignment.csv \
    --maps NOVAE_raw=eval/niches_xenium/domains_novae_raw.npy \
           NOVAE_joint=eval/niches_xenium/domains_clip_joint.npy \
           scConcept_joint=eval/niches_xenium_scconcept/domains_clip_joint.npy \
    --outdir results/eval/pathology_xenium
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path as PPath

import numpy as np
import pandas as pd
from matplotlib.path import Path

from eval_niches import load_views, load_obsm_aligned


def region_label(f):
    p = f.get("properties", {})
    c = p.get("classification")
    return (c.get("name") if isinstance(c, dict) else None) or p.get("name") or "NA"


def assign_regions(geojson, alignment_csv, coords, scale=0.2125):
    """Assigne une région de pathologie à chaque cellule (point-dans-polygone, en µm)."""
    M = np.loadtxt(alignment_csv, delimiter=",")
    feats = json.load(open(geojson)).get("features", [])
    polys = []
    for f in feats:
        ring = np.array(f["geometry"]["coordinates"][0], dtype=float)
        h = np.c_[ring, np.ones(len(ring))]
        polys.append((region_label(f), ((M @ h.T).T[:, :2]) * scale))
    # grandes régions d'abord → les petites (plus spécifiques) écrasent en cas de chevauchement
    polys.sort(key=lambda t: -Path(t[1]).get_extents().size.prod())
    region = np.array(["unannotated"] * len(coords), dtype=object)
    for name, poly in polys:
        region[Path(poly).contains_points(coords)] = name
    return region


def plot_overlay(coords, region, maps, outpath, max_points=60000, seed=0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  [info] matplotlib indisponible ({e})."); return
    rng = np.random.default_rng(seed)
    idx = np.arange(len(coords))
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    panels = [("Pathologie (vérité terrain)", region)] + [(k, v) for k, v in maps.items()]
    fig, ax = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), squeeze=False)
    for axi, (title, lab) in zip(ax[0], panels):
        codes = pd.Categorical(np.asarray(lab)[idx]).codes
        axi.scatter(coords[idx, 0], coords[idx, 1], c=codes, cmap="tab20", s=2, alpha=0.7)
        axi.set_aspect("equal"); axi.invert_yaxis(); axi.set_title(title)
        axi.set_xticks([]); axi.set_yticks([])
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)
    print(f"  carte superposée : {outpath}")


def main():
    ap = argparse.ArgumentParser(description="Validation des niches contre la pathologie (GeoJSON).")
    ap.add_argument("--paired-dir", required=True)
    ap.add_argument("--spatial-h5ad", required=True)
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--alignment", required=True, help="CSV matrice affine 3x3 (H&E alignment)")
    ap.add_argument("--scale", type=float, default=0.2125, help="µm/pixel (Xenium = 0.2125)")
    ap.add_argument("--maps", nargs="+", required=True, help="nom=domains.npy (2+ partitions)")
    ap.add_argument("--outdir", default="results/eval/pathology_xenium")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    out = PPath(a.outdir); out.mkdir(parents=True, exist_ok=True)
    _, _, cell_id, _ = load_views(a.paired_dir)
    coords, _ = load_obsm_aligned(a.spatial_h5ad, "spatial_um", cell_id, ndim=2)

    region = assign_regions(a.geojson, a.alignment, coords, a.scale)
    np.save(out / "path_region.npy", region)
    ann = region != "unannotated"
    print(f"cellules annotées : {ann.sum()}/{len(region)} ({100 * ann.mean():.1f}%)")
    print(pd.Series(region[ann]).value_counts().to_string())

    from sklearn.metrics import adjusted_rand_score, homogeneity_score, normalized_mutual_info_score

    maps, rows = {}, []
    for it in a.maps:
        assert "=" in it, f"--maps attend nom=chemin.npy, reçu {it!r}"
        name, path = it.split("=", 1)
        dom = np.load(path)
        assert len(dom) == len(region), f"{name}: {len(dom)} cellules != {len(region)}"
        maps[name] = dom
        ari = adjusted_rand_score(region[ann], dom[ann])
        nmi = normalized_mutual_info_score(region[ann], dom[ann])
        homo = homogeneity_score(region[ann], dom[ann])
        rows.append({"méthode": name, "ARI_vs_patho": round(ari, 4),
                     "NMI_vs_patho": round(nmi, 4), "homogénéité": round(homo, 4)})
        print(f"  {name:18s} ARI={ari:.4f} NMI={nmi:.4f} homog={homo:.4f}")

    table = pd.DataFrame(rows).sort_values("NMI_vs_patho", ascending=False)
    table.to_csv(out / "pathology_summary.csv", index=False)
    print("\n=== ACCORD NICHES ↔ PATHOLOGIE ===")
    print(table.to_string(index=False))
    print("ARI/NMI/homogénéité hauts = les niches retrouvent les régions du pathologiste.")
    if not a.no_plot:
        plot_overlay(coords, region, maps, out / "pathology_overlay.png")


if __name__ == "__main__":
    main()
