#!/usr/bin/env bash
# Balaye n_domains et mesure l'accord niches ↔ pathologie pour les 3 méthodes.
# Robustesse SANS biais : on rapporte tous les n (on ne choisit pas "le meilleur").
#
#   bash tools/scan_ndomains.sh xenium                 # n = 5 7 10 15
#   bash tools/scan_ndomains.sh xenium "5 8 10 15" cuda
#
# Sorties dans eval/scan_<dataset>/ : un dossier par n + scan_ndomains.png + scan_summary.csv.
# Les espaces clip_* nécessitent torch (ta machine). Sorties dédiées -> n'écrase rien.
set -euo pipefail

DATASET="${1:?usage: scan_ndomains.sh <xenium|cosmx> [\"5 7 10 15\"] [cpu|cuda]}"
NS="${2:-5 7 10 15}"
DEVICE="${3:-cpu}"

case "$DATASET" in
  xenium)
    PAIRED=data/processed/xenium_renal
    PAIRED_SCC=data/processed/xenium_renal_scconcept
    SPATIAL=data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad
    CKPT=runs/clip_xenium_seed42/best.pt
    CKPT_SCC=runs/clip_xenium_scconcept_seed42/best.pt
    GEO=data/raw/xenium_renal/supplemental/Xenium_V1_Human_Kidney_FFPE_Protein_updated_annotation.geojson
    ALIGN=data/raw/xenium_renal/supplemental/Xenium_V1_Human_Kidney_FFPE_Protein_updated_he_imagealignment.csv
    ;;
  *)
    echo "Seul 'xenium' a des annotations de pathologie (GeoJSON). Dataset reçu: $DATASET" >&2
    exit 1 ;;
esac

BASE="eval/scan_${DATASET}"
mkdir -p "$BASE"
echo "Scan n_domains = [$NS]  ->  $BASE"

for n in $NS; do
  echo "######## n_domains = $n ########"
  # niches NOVAE (novae_raw + clip_joint)
  python src/eval_niches.py --paired-dir "$PAIRED" --spatial-h5ad "$SPATIAL" --ckpt "$CKPT" \
    --spaces novae_raw,clip_joint --n-domains "$n" --device "$DEVICE" --no-plot \
    --outdir "$BASE/novae_n$n"
  # niches scConcept (clip_joint)
  python src/eval_niches.py --paired-dir "$PAIRED_SCC" --spatial-h5ad "$SPATIAL" --ckpt "$CKPT_SCC" \
    --spaces clip_joint --n-domains "$n" --device "$DEVICE" --no-plot \
    --outdir "$BASE/scc_n$n"
  # accord vs pathologie
  python src/pathology_validation.py --paired-dir "$PAIRED" --spatial-h5ad "$SPATIAL" \
    --geojson "$GEO" --alignment "$ALIGN" --no-plot \
    --maps NOVAE_raw="$BASE/novae_n$n/domains_novae_raw.npy" \
           NOVAE_joint="$BASE/novae_n$n/domains_clip_joint.npy" \
           scConcept_joint="$BASE/scc_n$n/domains_clip_joint.npy" \
    --outdir "$BASE/patho_n$n"
done

python3 tools/plot_ndomains_scan.py --base "$BASE" --ns "$NS"
echo "Terminé ✓  ->  $BASE/scan_ndomains.png"
