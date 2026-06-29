#!/usr/bin/env bash
# Pipeline complet d'évaluation des niches multi-omiques, pour un jeu de données.
#
#   bash tools/run_niches_pipeline.sh cosmx
#   bash tools/run_niches_pipeline.sh xenium
#
# Enchaîne (cf. src/README_niches.md) :
#   1. eval_niches  — modèle NOVAE   : novae_raw, clip_rna, clip_prot, clip_joint (+ FIDE/ARI/NMI)
#   2. eval_niches  — modèle scConcept : clip_joint
#   3. compare_niches — NOVAE vs scConcept (clip_joint) sur le même graphe
#   4. niche_information — info multi- vs uni-omique (variance expliquée + raffinement)
#
# Étapes 1-2 = ton modèle CLIP => torch requis (sur ta machine, pas dans le sandbox).
# Lance depuis la racine du repo. Adapte --device cpu/cuda et --n-domains au besoin.
set -euo pipefail

DATASET="${1:?usage: run_niches_pipeline.sh <cosmx|xenium>}"
NDOM="${2:-10}"        # nombre de niches (optionnel, défaut 10)
DEVICE="${3:-cpu}"     # cpu | cuda (optionnel)

case "$DATASET" in
  cosmx)
    PAIRED_NOVAE=data/processed/cosmx_breast
    PAIRED_SCC=data/processed/cosmx_breast_scconcept
    SPATIAL=data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad
    CKPT_NOVAE=runs/clip_cosmx_seed42/best.pt
    CKPT_SCC=runs/clip_cosmx_scconcept_seed42/best.pt
    ANNOT=annotation_out/cosmx_annotated.h5ad
    ;;
  xenium)
    PAIRED_NOVAE=data/processed/xenium_renal
    PAIRED_SCC=data/processed/xenium_renal_scconcept
    SPATIAL=data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad
    CKPT_NOVAE=runs/clip_xenium_seed42/best.pt
    CKPT_SCC=runs/clip_xenium_scconcept_seed42/best.pt
    ANNOT=annotation_out/xenium_annotated.h5ad
    ;;
  *) echo "Dataset inconnu : $DATASET (cosmx|xenium)"; exit 1 ;;
esac

# suffixe par n_domains (sauf 10) → ne JAMAIS écraser un autre n
SUF=""; [ "$NDOM" != "10" ] && SUF="_n${NDOM}"
OUT_NOVAE="eval/niches_${DATASET}${SUF}"
OUT_SCC="eval/niches_${DATASET}_scconcept${SUF}"
OUT_CMP="eval/compare_${DATASET}${SUF}"
OUT_INFO="eval/info_${DATASET}${SUF}"

echo "############ [$DATASET] 1/4 — eval_niches (modèle NOVAE) ############"
python src/eval_niches.py \
  --paired-dir "$PAIRED_NOVAE" --spatial-h5ad "$SPATIAL" --ckpt "$CKPT_NOVAE" \
  --labels-h5ad "$ANNOT" \
  --spaces novae_raw,clip_rna,clip_prot,clip_joint \
  --n-domains "$NDOM" --device "$DEVICE" --outdir "$OUT_NOVAE"

echo "############ [$DATASET] 2/4 — eval_niches (modèle scConcept) ############"
python src/eval_niches.py \
  --paired-dir "$PAIRED_SCC" --spatial-h5ad "$SPATIAL" --ckpt "$CKPT_SCC" \
  --labels-h5ad "$ANNOT" \
  --spaces clip_joint \
  --n-domains "$NDOM" --device "$DEVICE" --outdir "$OUT_SCC"

echo "############ [$DATASET] 3/4 — compare_niches (NOVAE vs scConcept) ############"
python src/compare_niches.py \
  --paired-dir "$PAIRED_NOVAE" --spatial-h5ad "$SPATIAL" --labels-h5ad "$ANNOT" \
  --maps NOVAE="$OUT_NOVAE/domains_clip_joint.npy" \
         scConcept="$OUT_SCC/domains_clip_joint.npy" \
  --outdir "$OUT_CMP"

echo "############ [$DATASET] 4/4 — niche_information (multi- vs uni-omique) ############"
python src/niche_information.py \
  --paired-dir "$PAIRED_NOVAE" \
  --joint-domains "$OUT_NOVAE/domains_clip_joint.npy" \
  --rna-domains  "$OUT_NOVAE/domains_novae_raw.npy" \
  --rna-target novae --n-domains "$NDOM" --outdir "$OUT_INFO"
# Pour un axe ARN totalement indépendant (plus lourd) :
#   ajoute  --rna-target counts --rna-counts-h5ad "$SPATIAL"

echo "############ [$DATASET] terminé. Résultats dans $OUT_NOVAE, $OUT_SCC, $OUT_CMP, $OUT_INFO ############"
