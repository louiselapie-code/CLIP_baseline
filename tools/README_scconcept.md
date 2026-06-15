# Annotation scConcept (remplace le Leiden) — mode d'emploi

But : produire une annotation de types cellulaires avec **scConcept** (conseil du tuteur),
pour remplacer le Leiden actuel dans l'évaluation du CLIP (ARI / NMI / sonde linéaire).

scConcept fournit des **embeddings** de cellules, pas des labels. L'« annotation » est donc :
**embeddings scConcept → clustering Leiden**. Pour les métriques (ARI/NMI/sonde) les clusters
suffisent ; on peut nommer les clusters plus tard (par gènes marqueurs) sans rien changer aux chiffres.

## Pourquoi sur Ruche et pas ici
scConcept exige **Python ≥ 3.12** et c'est un transformer de fondation → **GPU** pour ~600k cellules.
Le sandbox de cette session est CPU / Python 3.10 : la partie scConcept n'y tourne pas.
La partie clustering + sortie + intégration à l'éval, elle, est **validée localement** (self-test + run_clip).

## Fichiers (dans `tools/`)
- `scconcept_annotate.py` : embeddings scConcept → Leiden → `*_labels.csv` (+ `*_embedding.npy`).
- `run_scconcept_ruche.slurm` : job SLURM (à adapter : noms de partition/module Ruche).

## Étapes
1. **Copier sur Ruche** le projet (au moins `data/raw/*/h5ad/*_rna_with_spatial_split_seed42.h5ad`
   et `tools/`). `X` de ces .h5ad = comptes bruts (vérifié), `var_names` = symboles humains.
2. **Adapter** `run_scconcept_ruche.slurm` : `--partition`, `module load` (anaconda/cuda), `PROJECT`.
3. **Lancer** : `sbatch tools/run_scconcept_ruche.slurm`
   (ou en interactif : `python tools/scconcept_annotate.py --rna-h5ad <...> --out-prefix data/processed/<slide>/scconcept`).
4. **Sorties** : `data/processed/<slide>/scconcept_labels.csv` (colonnes `cell_id, split, scconcept_leiden`)
   + `scconcept_embedding.npy`. Le script imprime aussi un crosstab vs l'ancien Leiden (contrôle).

## Rebrancher dans l'évaluation
Ramener les petits `scconcept_labels.csv` dans le dossier projet, puis (CPU, ici ou sur Ruche) :

```
python tools/run_clip.py --src src --paired-dir data/processed/cosmx_breast \
  --outdir runs/clip_cosmx_seed42 --mode labels --eval-gallery 15000 --probe-train 15000 \
  --labels data/processed/cosmx_breast/scconcept_labels.csv --label-col scconcept_leiden
# idem pour xenium_renal
```

Je peux alors régénérer le rapport/slides avec ces nouveaux chiffres.

## À garder en tête
- **Circularité résiduelle** : scConcept reste basé sur l'ARN → un reste de circularité pour juger
  les espaces ARN. Mais scConcept ≠ NOVAE (modèle/entraînement différents), donc PAS trivialement
  circulaire comme le serait un clustering de NOVAE lui-même ; et c'est une méthode reconnue
  (endossée par ton tuteur). L'ancrage **sans label** (CLIP > CCA en retrieval + contrôle permuté)
  reste le plus fiable.
- **Résolution Leiden** (`--resolution`, défaut 1.0) : ↑ = plus de clusters. À ajuster selon le
  nombre de types voulu.
- **Modèle** : `corpus40M-model30M` (humain, défaut). Alternative multi-espèces : `corpus360M[multi-species]-model170M`.
