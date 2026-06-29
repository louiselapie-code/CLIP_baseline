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
- **ARI / NMI vs types** — recouvrement des niches avec `cell_type_final` (si `--labels-h5ad`).
  Score **bas = sain** (niches = voisinages, pas types cellulaires) ; score élevé = l'espace
  ne fait que du typage cellulaire. Dispo CosMx **et** Xenium.

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
- `--labels-h5ad <h5ad> --label-col cell_type_final` : ARI/NMI niches vs types cellulaires
  (exclut `--label-drop`, def. `incertaine`).
- `--cells all|train|test`, `--knn-graph k` (def. 6), `--max-fit-cells` (sous-échantillon KMeans).

Sortie supplémentaire si ≥ 2 espaces : `niches_compare.png` (cartes côte à côte).

## Comparer NOVAE vs scConcept (inter-runs) — `compare_niches.py`

Le `clip_joint` NOVAE et le `clip_joint` scConcept viennent de **deux runs** (checkpoints +
paired-dirs différents). On les compare avec `compare_niches.py`, qui aligne les
`domains_*.npy` par cell_id (ordre identique entre paired-dirs), les évalue sur le **même
graphe** et produit une figure côte à côte.

```bash
# 1) générer les deux cartes clip_joint
python src/eval_niches.py --paired-dir data/processed/cosmx_breast \
  --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
  --ckpt runs/clip_cosmx_seed42/best.pt --spaces clip_joint --n-domains 10 \
  --outdir eval/niches_cosmx_novae
python src/eval_niches.py --paired-dir data/processed/cosmx_breast_scconcept \
  --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
  --ckpt runs/clip_cosmx_scconcept_seed42/best.pt --spaces clip_joint --n-domains 10 \
  --outdir eval/niches_cosmx_scconcept
# 2) comparer
python src/compare_niches.py --paired-dir data/processed/cosmx_breast \
  --spatial-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad \
  --labels-h5ad annotation_out/cosmx_annotated.h5ad \
  --maps NOVAE=eval/niches_cosmx_novae/domains_clip_joint.npy \
         scConcept=eval/niches_cosmx_scconcept/domains_clip_joint.npy \
  --outdir eval/compare_cosmx
```

> ⚠️ Dans le paired-dir `*_scconcept`, `paired_rna.npy` **est** l'embedding scConcept (512d).
> L'espace `novae_raw` n'y a donc pas de sens — pour le modèle scConcept, n'utilise que
> `clip_*`. Pour le scConcept brut (non-CLIP), passe plutôt par `--spaces scconcept_raw`
> avec `--scconcept-h5ad` (sur le paired-dir NOVAE).

## Résultat de validation (CosMx, sous-échantillon, sanity check)

| espace          | dim | FIDE  | entropie_norm | heuristique | ARI_types | NMI_types |
|-----------------|-----|-------|---------------|-------------|-----------|-----------|
| `novae_raw`     | 64  | 0.673 | 0.971         | 0.654       | 0.023     | 0.063     |
| `scconcept_raw` | 512 | 0.354 | 0.782         | 0.277       | 0.053     | 0.167     |

→ Comme attendu, l'embedding **spatial** de NOVAE donne des niches bien plus continues
(FIDE 0.67) qu'un encodeur **par cellule** comme scConcept (FIDE 0.35) — même si scConcept
gagne au retrieval cellule-à-cellule. Et les niches scConcept collent davantage aux types
cellulaires (NMI 0.167 vs 0.063) : elles font plus du typage que de la niche. C'est
exactement la question à creuser pour `clip_joint`.

**Note** : associe le bon checkpoint au bon `--paired-dir` (NOVAE → rna 64 ; scConcept → rna 512).

## Les niches multi-omiques sont-elles plus *informatives* ? — `niche_information.py`

FIDE ne mesure que la **continuité spatiale**, pas l'information. Pour savoir si le
multi-omique apporte plus d'**information** que chaque modalité seule, on mesure la
**variance expliquée** (η², multivarié) de chaque modalité *brute* (protéine ; ARN =
comptages) par chaque partition de niches — même nb de domaines, mêmes cellules :

- une niche uni-omique explique bien SA modalité, mal l'autre ;
- une niche jointe vraiment intégrative explique (un peu moins, mais) les DEUX à la fois.

```bash
python src/niche_information.py \
  --paired-dir data/processed/cosmx_breast \
  --joint-domains eval/niches_cosmx/domains_clip_joint.npy \
  --rna-domains  eval/niches_cosmx/domains_novae_raw.npy \
  --rna-target novae --n-domains 10 --outdir eval/info_cosmx
# cible ARN INDÉPENDANTE (comptages bruts ; plus lourd, prévoir de la RAM) :
#   --rna-target counts --rna-counts-h5ad data/raw/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad
```

Sorties : `information_plane.png` (EV_ARN vs EV_protéine, un point par partition),
`contingency_*png` (comment une niche se scinde dans une autre), `refinement_example.png`
(une niche ARN scindée par la jointe + test de permutation montrant que les sous-niches
diffèrent en protéine), `information_summary.csv`, `information_report.json`.

**Résultat CosMx (n_domains=10, cible ARN = novae)** :

| partition | EV_ARN | EV_protéine |
|---|---|---|
| niche_ARN (uni) | 0.68 | 0.23 |
| niche_prot (uni) | 0.29 | 0.41 |
| **niche_joint** | 0.49 | 0.31 |
| aléatoire | ~0 | ~0 |

→ La niche jointe est un **intégrateur équilibré** : elle ne bat aucun spécialiste sur sa
propre modalité (budget de 10 domaines limité), mais c'est la seule partition correcte sur
les **deux**. Le zoom montre qu'une niche ARN se scinde en sous-niches aux profils
protéiques **réellement** distincts (permutation p≈0.005) → la protéine apporte une info que
l'ARN seul n'avait pas. Conclusion : information **jointe** en plus, pas information « par
modalité » en plus. (Cible ARN `novae` un peu circulaire pour l'axe ARN ; utilise `counts`
pour un axe ARN totalement indépendant.)
