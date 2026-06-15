#!/bin/bash
#SBATCH --job-name=scconcept_annot
#SBATCH --output=scconcept_%j.out
#SBATCH --error=scconcept_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# ⚠️ ADAPTER à Ruche : nom de partition GPU (ex. --partition=gpu / gpua100), compte, etc.
#SBATCH --partition=gpu

set -e

# ----------------------------------------------------------------------------
# 0) Variables à adapter
# ----------------------------------------------------------------------------
PROJECT=$HOME/CLIP_baseline            # racine du projet sur Ruche (où sont data/ et tools/)
ENV_NAME=scconcept                     # nom de l'env conda
RES=1.0                                # résolution Leiden (↑ = plus de clusters)

cd "$PROJECT"

# ----------------------------------------------------------------------------
# 1) Environnement Python >= 3.12 (scConcept l'exige)
#    Option A (conda) :
# ----------------------------------------------------------------------------
module purge
module load anaconda3 2>/dev/null || true     # ⚠️ ADAPTER au nom du module sur Ruche
module load cuda      2>/dev/null || true      # ⚠️ ADAPTER (cuda/12.x)

if ! conda env list | grep -q "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
source activate "$ENV_NAME" || conda activate "$ENV_NAME"
python -c "import concept" 2>/dev/null || pip install -q sc-concept scanpy leidenalg igraph
# (optionnel, accélère) : MAX_JOBS=4 pip install "flash-attn>=2.7" --no-build-isolation

# Option B (uv) — décommenter si tu préfères uv :
# module load uv 2>/dev/null || true
# uv venv --python 3.12 .venv_scconcept && source .venv_scconcept/bin/activate
# uv pip install sc-concept scanpy leidenalg igraph

# ----------------------------------------------------------------------------
# 2) Annotation scConcept des deux slides
#    X des .h5ad ARN = comptes bruts (vérifié), var_names = symboles humains.
# ----------------------------------------------------------------------------
python tools/scconcept_annotate.py \
  --rna-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
  --out-prefix data/processed/cosmx_breast/scconcept \
  --model corpus40M-model30M --species hsapiens --resolution $RES \
  --cache-dir "$PROJECT/scconcept_cache"

python tools/scconcept_annotate.py \
  --rna-h5ad data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
  --out-prefix data/processed/xenium_renal/scconcept \
  --model corpus40M-model30M --species hsapiens --resolution $RES \
  --cache-dir "$PROJECT/scconcept_cache"

echo "Terminé. Labels : data/processed/<slide>/scconcept_labels.csv"
