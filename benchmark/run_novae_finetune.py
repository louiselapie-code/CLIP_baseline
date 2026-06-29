"""
run_novae_finetune.py — Baseline NOVAE end-to-end (FINE-TUNÉ), ARN seul.

NOVAE entraîné sur ta slide en auto-supervisé (aucun label) : on part du modèle pré-entraîné
et on le ré-entraîne quelques epochs, puis on assigne les domaines. C'est une baseline NOVAE
plus forte que le zero-shot (= ton `novae_raw`).

⚠️ Nécessite torch + le package `novae` (ta machine — non testé dans le sandbox).
Reproduit les conventions de src/precompute_novae_embeddings.py : comptes bruts dans X,
`novae.spatial_neighbors`, MÊME `model_name` et MÊME `radius` qu'à ton précalcul.

Sortie : domains_novae_finetuned.npy aligné sur l'ordre des cellules de --paired-dir,
à brancher sur src/pathology_validation.py --maps.

Exemple :
  python benchmark/run_novae_finetune.py \
    --rna-h5ad data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
    --paired-dir data/processed/xenium_renal \
    --model MICS-Lab/novae-human-0 --radius <le même qu'au précalcul> \
    --n-domains 5 --max-epochs 10 --accelerator cpu --outdir eval/benchmark_xenium
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_niches import load_views  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Baseline NOVAE fine-tuné (ARN seul).")
    ap.add_argument("--rna-h5ad", required=True, help="h5ad ARN comptes bruts + obsm['spatial']")
    ap.add_argument("--paired-dir", required=True, help="pour aligner par cell_id")
    ap.add_argument("--model", default="prism-oncology/novae-human-0", help="même model_name qu'au précalcul")
    ap.add_argument("--radius", type=float, default=50.0, help="même radius (µm) qu'au précalcul")
    ap.add_argument("--slide-key", default=None)
    ap.add_argument("--n-domains", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--outdir", default="eval/benchmark_xenium")
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    import anndata as ad
    import novae

    adata = ad.read_h5ad(a.rna_h5ad)
    assert "spatial" in adata.obsm, "obsm['spatial'] requis (comme au précalcul NOVAE)."
    novae.spatial_neighbors(adata, slide_key=a.slide_key, radius=a.radius)

    model = novae.Novae.from_pretrained(a.model)
    # fine-tuning AUTO-SUPERVISÉ sur la slide (le signal vient des paires de sous-graphes voisins)
    model.fine_tune(adata, max_epochs=a.max_epochs, lr=a.lr,
                    accelerator=a.accelerator, num_workers=a.num_workers)
    model.compute_representations(adata, accelerator=a.accelerator, num_workers=a.num_workers)
    key = model.assign_domains(adata, n_domains=a.n_domains)
    print("clé domaines NOVAE :", key)

    # aligne adata.obs[key] sur l'ordre des cellules appariées
    _, _, cell_id, _ = load_views(a.paired_dir)
    ids = adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs.columns else adata.obs_names.astype(str)
    mp = dict(zip(ids.astype(str), adata.obs[key].astype(str)))
    raw = np.array([mp.get(str(c), "NA") for c in cell_id], dtype=object)
    uniq = {d: i for i, d in enumerate(sorted(set(raw) - {"NA", "nan"}))}
    dom = np.array([uniq.get(d, -1) for d in raw], dtype=np.int64)
    np.save(out / "domains_novae_finetuned.npy", dom)
    print(f"écrit {out/'domains_novae_finetuned.npy'} ({len(uniq)} domaines, {(dom < 0).sum()} non assignées)")


if __name__ == "__main__":
    main()
