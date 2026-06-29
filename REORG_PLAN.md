# Plan de réorganisation — repo CLIP_baseline

> Rédigé le 2026-06-29. **Rien n'a été exécuté.** À appliquer dans une session dédiée,
> après tes réponses aux questions en bas. Aucune suppression ne sera faite sans ton feu vert explicite.

## Principe
Le poids du repo (~40 Go) = tes **vraies données** (`data/` 26 Go + `annotation_out/` 13 Go, surtout les `.h5ad`).
Supprimer ne libère quasi rien : l'objectif est la **structure**, pas le gain de place.
On range, on renomme de façon cohérente, on regroupe les sorties, et on ne supprime QUE les variantes
d'expériences que tu confirmes obsolètes.

## Structure cible proposée
```
CLIP_baseline/
├── README.md                  # à créer : vue d'ensemble + comment lancer
├── .gitignore                 # ajouter __pycache__/, *.pyc, outputs/
├── config/                    # inchangé
├── src/                       # code modèle + éval (retirer __pycache__/)
├── tools/                     # scripts utilitaires (retirer tools/outputs/)
├── benchmark/                 # baselines (banksy, CN, novae finetune)
├── data/
│   ├── raw/<slide>/{h5ad,tables,figures}      # inchangé
│   └── processed/<slide>/                     # 1 dossier par slide (variantes à trancher)
├── annotation/                # ex-"annotation_out/"
│   ├── <slide>_annotated.h5ad
│   └── labels/
│       ├── <slide>_labels.csv                 # version propre (cell_id, cell_type)
│       └── <slide>_labels_full.csv            # version complète (pred, conf, final)
├── results/
│   ├── runs/                  # checkpoints + courbes (ex "runs/")
│   ├── eval/                  # sorties d'éval par expérience (ex "eval/")
│   └── reports/               # ex "report/" + "reports_2026-06-08/" fusionnés
└── notebook/
```

## A. Renommages / déplacements (sûrs, réversibles via git mv)
| Actuel | Cible |
|---|---|
| `annotation_out/` | `annotation/` |
| `annotation_out/labels_cosmx.csv` | `annotation/labels/cosmx_labels.csv` |
| `annotation_out/labels_xenium.csv` | `annotation/labels/xenium_labels.csv` |
| `annotation_out/cosmx_labels.csv` | `annotation/labels/cosmx_labels_full.csv` |
| `annotation_out/xenium_labels.csv` | `annotation/labels/xenium_labels_full.csv` |
| `annotation_out/create_csv.py` | `tools/make_clean_labels.py` (renommé, explicite) |
| `runs/` | `results/runs/` |
| `eval/` | `results/eval/` |
| `report/` | `results/reports/` |
| `reports_2026-06-08/` | `results/reports/2026-06-08/` (snapshot daté conservé) |

**Cohérence de nommage** : tout passer en `<slide>_<contenu>` (aujourd'hui on a les deux ordres,
ex. `labels_cosmx.csv` ET `cosmx_labels.csv`). Idem pour les dossiers d'expériences (voir §C).

## B. Suppressions CANDIDATES — à confirmer (je ne supprime rien sans ton OK)
- `outputs/` (racine) — **vide**.
- `src/__pycache__/` et `tools/outputs/` — artefacts/égarés (et à mettre en `.gitignore`).
- Variantes d'expériences obsolètes — **à trancher par toi** (§C).

## C. Variantes d'expériences — j'ai besoin de ton arbitrage
Plusieurs familles de suffixes cohabitent. Dis-moi pour chacune : **garder / archiver / supprimer**.

- **`_scconcept`** (`eval/*_scconcept`, `data/processed/*_scconcept`, `runs/*_scconcept_seed42`)
  → a priori **les finales** (annotation scConcept). À garder ?
- **`_seed42`** (`runs/clip_*_seed42`) → runs de référence. À garder ?
- **`_c`** (`data/processed/{cosmx_c,joint_c,xenium_c}`, `runs/{clip_cosmx_c,clip_joint_c,clip_xenium_c}`)
  → quoi exactement ? (variante « counts » ? debug ?) Obsolètes ?
- **`_counts`** (`eval/info_cosmx_counts`, `eval/info_xenium_counts`) vs `eval/info_xenium` → doublon ?
- **`_n5`** (`eval/niches_xenium_n5`, `eval/pathology_xenium_n5`) → réglage de paramètre ; garder un seul ?
- **`_newannot`** (`eval/cosmx_newannot`, `eval/xenium_newannot`) vs `_scconcept` → intermédiaire à jeter ?
- **divers runs** : `_smoketest_cosmx`, `clip_cosmx_bs1024`, `clip_joint_2tower` → tests ? supprimables ?

## D. Exécution (session dédiée)
1. Créer la nouvelle arbo + `git mv` (renommages/déplacements §A) → **aucune perte**.
2. Mettre à jour les chemins en dur dans les scripts (`tools/*.py`, `src/*` pointant vers `eval/`, `runs/`, `report/`).
3. Suppressions §B + §C **uniquement** sur les éléments que tu as validés (via la confirmation de suppression).
4. Ajouter `.gitignore` (`__pycache__/`, `*.pyc`, `outputs/`) + un `README.md` de structure.
5. Commit clair : `chore: réorganisation et nettoyage de l'arborescence`.

## À décider avant de lancer
- Les arbitrages du §C (familles de variantes).
- Garde-t-on les 2 versions de labels (`*_labels.csv` propre **et** `*_labels_full.csv`) ? (recommandé : oui)
- Les gros `.h5ad` (`*_annotated.h5ad`, `data/raw/.../*.h5ad`) : rester dans le repo, ou sortir du suivi git
  (ils sont énormes ; souvent on les met hors-git / DVC / stockage séparé) ?
