# Niches multi-omiques via la « fin de NOVAE »

Transforme des embeddings cellulaires (sortie CLIP, ou NOVAE/scConcept bruts) en **niches
spatiales**, en réutilisant la tête de niches de NOVAE — sans ré-entraîner NOVAE.

## Deux fichiers

- **`niches.py`** — la tête NOVAE portée (numpy + scikit-learn, dépendances légères) :
  1. projection L2 sur la sphère unité ;
  2. prototypes par **KMeans** (K=512 par défaut) ;
  3. attribution d'un prototype « feuille » par cellule (**argmax** cosinus = inférence NOVAE ;
     variante **Sinkhorn**-OT dispo) ;
  4. **clustering hiérarchique** des prototypes (cosine, average) → arbre ;
  5. coupe de l'arbre à `n_domains`/`level` → niches (ou **Leiden** sur les prototypes).
- **`eval_niches.py`** — le driver : charge embeddings + coords spatiales, calcule les niches
  sur plusieurs espaces et les évalue avec les métriques NOVAE.

> ⚠️ **Sinkhorn ≠ inférence.** Dans NOVAE, le transport optimal (Sinkhorn-Knopp) sert pendant
> l'**entraînement** (cible de la loss SwAV). À l'**inférence**, l'attribution se fait par
> argmax cosinus. Par défaut on reproduit l'inférence (`--assign argmax`). `--assign sinkhorn`
> est une variante (plus proche de la description « transport optimal », mais non utilisée par
> NOVAE à l'inférence).

## Espaces d'embeddings comparés (`--spaces`)

| nom             | embedding                          | spatial ? |
|-----------------|------------------------------------|-----------|
| `novae_raw`     | `paired_rna.npy` (NOVAE gelé)      | oui (GNN) — réf. « NOVAE seul » |
| `clip_rna`      | `z_r = rna_head(NOVAE)`            | oui (via NOVAE) |
| `clip_prot`     | `z_p = protein_tower(prot)`        | non (par cellule) |
| `clip_joint`    | `l2(z_r + z_p)` — **multi-omique** | partiel — **l'espace clé** |
| `scconcept_raw` | `obsm['X_scConcept']` d'un .h5ad   | non (par cellule) |

Tous les espaces sont évalués sur le **même graphe spatial** → comparaison équitable.

## Métriques (port de `novae/monitor/eval.py`)

- **FIDE** — F1 des arêtes intra-domaine. **Haut = niches spatialement continues.**
- **entropie normalisée** — équilibre des tailles de niches (haut = équilibré).
- **heuristique** — `FIDE × entropie_norm` (compromis continuité / diversité).
- **JSD** — divergence inter-slides (bas = bon mélange). Calculé seulement si `--slide-key`
  pointe une colonne `obs` avec ≥ 2 slides (sinon `None`).

## Exemples

```bash
# NOVAE : ARN seul vs CLIP multi-omique (CosMx)
python src/eval_niches.py \
  --paired-dir data/processed/cosmx_breast \
  --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
  --ckpt runs/clip_cosmx_seed42/best.pt \
  --spaces novae_raw,clip_rna,clip_prot,clip_joint \
  --n-domains 10 --outdir eval/niches_cosmx

# scConcept : même comparaison (⚠️ paired-dir scconcept, rna_dim=512)
python src/eval_niches.py \
  --paired-dir data/processed/cosmx_breast_scconcept \
  --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
  --ckpt runs/clip_cosmx_scconcept_seed42/best.pt \
  --spaces clip_joint --n-domains 10 --outdir eval/niches_cosmx_scconcept

# Ajouter scConcept brut comme comparaison d'encodeur ARN (Xenium)
python src/eval_niches.py \
  --paired-dir data/processed/xenium_renal \
  --spatial-h5ad data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
  --ckpt runs/clip_xenium_seed42/best.pt \
  --scconcept-h5ad annotation_out/xenium_annotated.h5ad \
  --spaces novae_raw,clip_joint,scconcept_raw --n-domains 10 --outdir eval/niches_xenium
```

Sorties par `--outdir` : `niches_summary.csv`, `niches_report.json`, `domains_<espace>.npy`,
`niches_<espace>.png` (carte spatiale).

## Options utiles

- `--num-prototypes` (def. 512), `--n-domains` (def. 10) ou `--level`, `--niche-method leiden --resolution R`.
- `--assign sinkhorn` : variante OT à l'attribution.
- `--smooth-knn K` : moyenne l'embedding sur K voisins spatiaux **avant** les prototypes —
  utile pour les espaces purement cellulaires (`clip_prot`, `scconcept_raw`) afin d'obtenir des
  niches plutôt que des types cellulaires.
- `--cells all|train|test`, `--knn-graph k` (def. 6), `--max-fit-cells` (sous-échantillon KMeans).

## Résultat de validation (CosMx, sous-échantillon, sanity check)

| espace          | dim | FIDE  | entropie_norm | heuristique |
|-----------------|-----|-------|---------------|-------------|
| `novae_raw`     | 64  | 0.673 | 0.971         | 0.654       |
| `scconcept_raw` | 512 | 0.354 | 0.782         | 0.277       |

→ Comme attendu, l'embedding **spatial** de NOVAE donne des niches bien plus continues
(FIDE 0.67) qu'un encodeur **par cellule** comme scConcept (FIDE 0.35) — même si scConcept
gagne au retrieval cellule-à-cellule. C'est exactement la question à creuser pour `clip_joint`.

**Note** : associe le bon checkpoint au bon `--paired-dir` (NOVAE → rna 64 ; scConcept → rna 512).
