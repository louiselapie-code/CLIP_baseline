# Partie Protéines — CLIP ARN–protéine

Implémentation des §2.3 (préprocessing + encodeur MLP) et §2.4 (tête de projection) de la stratégie.
Miroir du côté ARN (`precompute_novae_embeddings.py`).

## Fichiers

- `preprocess_protein.py` — pipeline de préparation des intensités d'anticorps.
- `protein_encoder.py` — `ProteinEncoder` (MLP shallow) + `ProteinProjectionHead` (Linear+L2) + `ProteinTower` (les deux combinés). C'est `ProteinTower` qui se branche dans le modèle CLIP.
- `make_synthetic_protein_data.py` — génère un `.h5ad` CosMx-like pour tester sans données réelles.

## Pipeline de préprocessing (§2.3)

`exclusion canaux techniques → clip p1–p99 → log(1+x) → standardisation (mean=0, std=1)`

**Les stats (bornes de clip, mean, std) sont ajustées sur `split == "train"` uniquement**, puis
appliquées à val/test, et sauvegardées en JSON pour l'inférence. Le doc ne le précisait pas, mais
sans ça il y a fuite de données et l'éval est faussée (d'autant plus avec le décalage technique
connu du bloc test Xenium).

## Test rapide (sans données réelles)

```bash
python make_synthetic_protein_data.py --out syn_prot.h5ad --n-cells 3000
python preprocess_protein.py --inputs syn_prot.h5ad --outdir features \
    --exclude IgG1 IgG2a IgG2b Negative DAPI
python protein_encoder.py          # smoke test du modèle
```

## Run réel

```bash
python preprocess_protein.py \
    --inputs cosmx_protein_with_spatial_split_seed42.h5ad \
    --outdir protein_features \
    --exclude <TES_CANAUX_TECHNIQUES>      # liste EXACTE à fournir (ou --exclude-file liste.txt)
```

Puis côté entraînement :

```python
from protein_encoder import ProteinTower
import numpy as np
X = np.load("protein_features/cosmx_protein_..._protein_features.npy")  # (N, P)
tower = ProteinTower(in_dim=X.shape[1])   # P inféré du panel ; dproj=256 comme le côté ARN
z_p = tower(torch.from_numpy(X).float())  # (N, 256), normalisé L2
```

## Sorties par fichier (dans `--outdir`)

| Fichier | Contenu |
|---|---|
| `{stem}_protein_features.npy` | matrice `(N, P)` float32, entrée du `ProteinTower` |
| `{stem}_cells.csv` | `cell_id, slide_id, split` — **même schéma et même ordre que le côté ARN** → réalignement ARN↔protéine par `cell_id` |
| `{stem}_protein_norm_stats.json` | marqueurs gardés + bornes clip + mean/std (train) pour réappliquer à l'identique |

## Choix par défaut (modifiables) et points à confirmer

- **Liste d'exclusion** : à remplir (`--exclude` ou `--exclude-file`). Si vide → tous les canaux gardés + warning.
- **Stats par fichier (par slide)** : CosMx et Xenium ont des panels/échelles différents → chacun ses propres stats. C'est le défaut robuste.
- **`P` inféré du panel** après exclusion (le « vecteur dim 64 » du doc n'est qu'un exemple).
- **`log1p` suppose des intensités brutes ≥ 0** ; si tes données sont déjà transformées, passe `--no-log1p`.
- **Tête de projection sans biais** (`proj_bias=False`, standard CLIP) ; modifiable.
- **Noms de colonnes** : `--split-col split`, `--slide-col slide_id`, `--id-col` (défaut `obs_names`).
  → **à aligner sur les conventions de ton `precompute_novae_embeddings.py`** (je n'ai pas eu accès au fichier).

## Hyperparamètres (défauts = tableau §5.1)

`hidden_dim=128, latent_dim=128, dproj=256, depth=2, dropout=0.1`, projection sans activation + L2.
`depth` configurable ∈ {1, 2, 3}. Fixer la seed via `protein_encoder.set_seed(...)` avant de construire le modèle.
