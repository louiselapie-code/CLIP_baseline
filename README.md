# CLIP_baseline

Modèle CLIP-like alignant **transcriptomique spatiale (ARN)** et **protéomique spatiale**
sur deux slides à cellules strictement appariées (CosMx ↔ Xenium). Premier modèle baseline.

## Structure du dépôt

```
CLIP_baseline/
├── config/                  # hyperparamètres / configs
├── src/                     # modèle CLIP + entraînement + évaluation
├── tools/                   # scripts utilitaires (figures, recaps, labels propres)
├── benchmark/               # baselines (BANKSY, cellular neighborhoods, NOVAE finetune)
├── data/
│   ├── raw/<slide>/         # h5ad bruts, tables, figures        (hors git)
│   ├── interim/
│   └── processed/<slide>/   # vues prétraitées par slide         (hors git)
├── annotation/
│   ├── <slide>_annotated.h5ad        # h5ad annotés               (hors git)
│   └── labels/
│       ├── <slide>_labels.csv        # version propre (cell_id, cell_type)
│       └── <slide>_labels_full.csv   # version complète (pred, conf, final)
├── results/
│   ├── runs/                # checkpoints + courbes d'entraînement
│   ├── eval/                # sorties d'évaluation par expérience
│   └── reports/             # rapports (+ snapshot daté 2026-06-08/)
└── notebook/
```

Les données lourdes (`data/`, `*.h5ad`, `*.npy`, `*.pt`, …) sont **suivies hors git**
(voir `.gitignore`) : seuls le code et les sorties légères (CSV, PNG, JSON) sont versionnés.

## Points d'entrée

Chaque script porte un exemple d'invocation dans sa docstring (chemins mis à jour vers `results/…`).

| Script | Rôle |
|---|---|
| `src/train.py` | Entraînement du modèle CLIP (`--paired-dir`, `--outdir`, défaut `results/runs/baseline`) |
| `src/evaluate.py` | Évaluation (retrieval, UMAP), défaut `results/eval` |
| `src/eval_niches.py` | Niches spatiales à partir des embeddings, défaut `results/eval/niches` |
| `src/compare_niches.py` | Comparaison de niches entre espaces (ARN / protéine / joint) |
| `src/niche_information.py` | Plan d'information ARN ↔ protéine |
| `src/pathology_validation.py` | Validation vis-à-vis de la pathologie |
| `src/eval_intra_type.py` | Retrieval intra-type cellulaire |
| `src/sweep.py` | Balayage d'hyperparamètres |
| `tools/make_clean_labels.py` | Génère `annotation/labels/<slide>_labels.csv` depuis les h5ad annotés |
| `benchmark/run_*.py` | Baselines comparatives (BANKSY, CN, NOVAE finetune) |

## Convention de nommage

Dossiers et fichiers suivent `<slide>_<contenu>` (ex. `cosmx_labels.csv`, `xenium_annotated.h5ad`).
Les expériences de `results/` portent un suffixe explicite : `_seed42` (référence),
`_scconcept` (annotation scConcept, finales), `_counts` (ARN counts brut, ≠ vue NOVAE).
