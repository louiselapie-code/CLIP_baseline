"""
precompute_novae_finetuned.py — Embeddings ARN NOVAE FINE-TUNÉS, au format de ton précalcul.

Identique à src/precompute_novae_embeddings.py, MAIS on fine-tune NOVAE sur la slide avant
d'extraire les représentations. Sortie au MÊME format (réutilise `save_embeddings`) → drop-in
pour reconstruire la vue ARN du CLIP, puis ré-entraîner le CLIP sur de l'ARN fine-tuné.

But : obtenir la comparaison ÉQUITABLE au niveau fine-tuné — multi-omique(ARN fine-tuné)
vs NOVAE fine-tuné (ARN seul), même encodeur des deux côtés → isole l'apport protéine.

⚠️ torch + novae requis (ta machine, non testé ici). MÊME --model / --radius qu'au précalcul
(prism-oncology/novae-human-0, radius 50). output_size reste 64 → config CLIP inchangée.

Exemple :
  python benchmark/precompute_novae_finetuned.py \
    --inputs data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
    --model prism-oncology/novae-human-0 --radius 50 --max-epochs 10 \
    --outdir data/interim/novae_finetuned_xenium
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from precompute_novae_embeddings import save_embeddings  # noqa: E402  (même format que ton précalcul)


def main():
    ap = argparse.ArgumentParser(description="Précalcul des embeddings NOVAE FINE-TUNÉS.")
    ap.add_argument("--inputs", nargs="+", required=True, help=".h5ad ARN comptes bruts + obsm['spatial']")
    ap.add_argument("--model", default="prism-oncology/novae-human-0")
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--slide-key", default=None)
    ap.add_argument("--split-key", default="split")
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--outdir", default="data/interim/novae_finetuned")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    import anndata as ad
    import novae

    for path in a.inputs:
        print(f"\n=== {path} ===")
        adata = ad.read_h5ad(path)
        assert "spatial" in adata.obsm, f"obsm['spatial'] requis dans {path}"
        novae.spatial_neighbors(adata, slide_key=a.slide_key, radius=a.radius)

        model = novae.Novae.from_pretrained(a.model)
        model.fine_tune(adata, max_epochs=a.max_epochs, lr=a.lr,
                        accelerator=a.accelerator, num_workers=a.num_workers)
        # représentations DU modèle fine-tuné -> obsm['novae_latent']
        model.compute_representations(adata, accelerator=a.accelerator, num_workers=a.num_workers)

        emb = np.asarray(adata.obsm["novae_latent"])
        cell_ids = (adata.obs["cell_id"].astype(str).tolist()
                    if "cell_id" in adata.obs.columns else adata.obs_names.astype(str).tolist())
        stem = os.path.splitext(os.path.basename(path))[0]
        slide_ids = (adata.obs[a.slide_key].astype(str).tolist()
                     if (a.slide_key and a.slide_key in adata.obs) else [stem] * adata.n_obs)
        splits = (adata.obs[a.split_key].astype(str).tolist()
                  if (a.split_key and a.split_key in adata.obs) else None)
        paths = save_embeddings(emb, cell_ids, slide_ids, a.outdir, stem, splits=splits)
        print(f"  embeddings fine-tunés {emb.shape} → {paths}")

    print("\nOK — utilise ces embeddings comme vue ARN pour reconstruire le paired-dir, puis ré-entraîne le CLIP.")


if __name__ == "__main__":
    main()
