# Benchmark — niches multi-omiques vs méthodes de domaines spatiaux

But : comparer tes niches (NOVAE_joint, NOVAE_raw, scConcept_joint) aux **méthodes de
référence** de domaines spatiaux (ARN seul), contre la **vérité terrain pathologie** (Xenium).
C'est ce qui manque pour rendre le résultat publiable (cf. l'article NOVAE, qui compare à
SpaceFlow/GraphST/STAGATE/SEDR).

## Principe (ce qui se fait en pratique)

Chaque méthode est une **boîte noire** : elle prend les données spatiales et produit **un
label de domaine par cellule**. Toutes les méthodes exportent un fichier standardisé :

    domains_<méthode>.npy   # entiers, aligné sur l'ordre des cellules de --paired-dir

Ensuite, **une seule** évaluation compare toutes les méthodes au même endroit, au même
`n_domains`, avec les mêmes métriques :

    python src/pathology_validation.py \
      --paired-dir data/processed/xenium_renal \
      --spatial-h5ad data/raw/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad \
      --geojson  data/raw/xenium_renal/supplemental/..._annotation.geojson \
      --alignment data/raw/xenium_renal/supplemental/..._he_imagealignment.csv \
      --maps NOVAE_joint=eval/scan_xenium/novae_n5/domains_clip_joint.npy \
             NOVAE_raw=eval/scan_xenium/novae_n5/domains_novae_raw.npy \
             scConcept_joint=eval/scan_xenium/scc_n5/domains_clip_joint.npy \
             BANKSY_style=eval/benchmark_xenium/domains_banksy.npy \
             NOVAE_finetune=eval/benchmark_xenium/domains_novae_finetuned.npy \
      --outdir eval/benchmark_xenium/patho_n5

Sorties dans `eval/benchmark_<dataset>/`.

## Runners disponibles

| runner | méthode | torch ? | testé |
|---|---|---|---|
| `run_banksy.py` | BANKSY-style (propre + voisinage, λ-pondéré, KMeans) | non | ✅ tourné |
| `run_novae_finetune.py` | NOVAE fine-tuné (end-to-end, auto-supervisé) | oui | ⚠️ à lancer chez toi |

`run_banksy.py` est une version **transparente, sans le filtre de Gabor azimutal (AGF)** du
package officiel `banksy_py`. Pour un résultat « officiel » (publication), installe
`banksy_py` et remplace ce runner.

## Pour aller plus loin (publication)

Méthodes de référence ARN-seul à ajouter (chacune doit exporter un `domains_<méthode>.npy`
aligné) : **BANKSY officiel**, **STAGATE**, **GraphST**, éventuellement **SEDR**, **SpaGCN**.
Ce sont des auto-encodeurs de graphe (torch) → plus lourds, à lancer sur ta machine.
Le cadre ci-dessus (export standardisé + `pathology_validation --maps`) les accueille tels quels.

## Résultat actuel (Xenium, n=5, vs pathologie)

| méthode | ARI | NMI |
|---|---|---|
| **NOVAE_joint (multi-omique)** | **0.43** | **0.34** |
| BANKSY-style (ARN) | 0.19 | 0.28 |
| NOVAE_raw (ARN) | 0.19 | 0.16 |
| scConcept_joint | 0.04 | 0.12 |

→ Le multi-omique bat les baselines ARN-seul (dont une vraie méthode de domaines spatiaux).
À confirmer avec BANKSY officiel / STAGATE / GraphST + plus de slides.
