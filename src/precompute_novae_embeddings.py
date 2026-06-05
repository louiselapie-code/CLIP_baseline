#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
 NOVAE — précalcul des embeddings ARN (une seule fois, en LOCAL sur CPU)
============================================================================

NOVAE est GELÉ (zero-shot) : on ne l'entraîne pas, on ne fait qu'une passe
avant pour extraire les embeddings de chaque cellule. Aucune dépendance CUDA
(contrairement à scConcept) -> ça tourne sur ton Mac. On le lance UNE fois,
on sauvegarde les embeddings sur disque, et ensuite tout l'entraînement CLIP
réutilise ces embeddings en cache (instantané).

Ce que fait le script :
  1. charge ton/tes fichier(s) .h5ad ARN (comptes BRUTS dans adata.X) ;
  2. construit le graphe spatial (novae.spatial_neighbors) ;
  3. charge le modèle pré-entraîné gelé (Novae.from_pretrained) ;
  4. calcule les représentations en mode zero-shot, sur CPU ;
  5. sauvegarde, par slide : les embeddings (.npy), l'index des cellules
     (.csv, pour réaligner avec la protéine) et, en option, un .h5ad complet.

Les embeddings sont rangés par NOVAE dans `adata.obsm["novae_latent"]`.

------------------------------------------------------------------
 Prérequis
------------------------------------------------------------------
    pip install novae          # installe aussi torch/torch_geometric/lightning (CPU OK)
  -> le 1er lancement TÉLÉCHARGE le modèle depuis HuggingFace (besoin d'internet
     une seule fois ; ensuite c'est en cache).

------------------------------------------------------------------
 Points IMPORTANTS pour TES données (CosMx / Xenium)
------------------------------------------------------------------
  * adata.X doit contenir les COMPTES BRUTS. NOVAE normalise lui-même
    (normalize_total + log1p) et garde les bruts dans adata.layers['counts'].
    -> ne passe PAS de données déjà normalisées/scalées.
  * adata.obsm['spatial'] doit exister (tes deux slides l'ont).
  * UNITÉS des coordonnées : NOVAE raisonne en MICRONS. Si tes coords sont en
    PIXELS, règle SCALE_TO_MICRONS (CosMx ≈ 0.1203, Xenium ≈ 0.2125 px→µm).
    --> Si tu pars de la SORTIE de ton notebook de split (OVERWRITE_SPATIAL_WITH_UM),
        obsm['spatial'] est DÉJÀ en µm : laisse scale_to_microns=None.
  * ORDRE vs SPLIT (important) : donne à NOVAE la slide COMPLÈTE
    (`*_with_spatial_split.h5ad`, qui contient TOUTES les cellules, buffer inclus),
    PAS la version `*_no_buffer.h5ad`. NOVAE a besoin du voisinage spatial intact ;
    on filtre par `obs['split']` APRÈS le calcul des embeddings (le script recopie
    la colonne `split` dans le .csv pour faciliter ça).
  * Si UN fichier contient PLUSIEURS slides (ex. CosMx + Xenium concaténés),
    renseigne SLIDE_KEY (colonne de adata.obs, ex. 'slide_ID') pour que le
    graphe soit construit slide par slide. Auto-détection sinon.
  * Ce script concerne la slide ARN (panel de gènes). La protéine (69 marqueurs)
    ne passe PAS par NOVAE : elle sera utilisée directement côté CLIP.

------------------------------------------------------------------
 Lancer
------------------------------------------------------------------
    python precompute_novae_embeddings.py \
        --inputs chemin/CosMx_rna.h5ad chemin/Xenium_rna.h5ad \
        --outdir novae_embeddings --radius 100

    python precompute_novae_embeddings.py --selftest   # teste la logique de sauvegarde
============================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ===========================================================================
# CONFIGURATION (modifiable ici, ou via les arguments en ligne de commande)
# ===========================================================================
@dataclass
class Config:
    # Chemin(s) vers le(s) .h5ad ARN (comptes bruts) = SORTIE de ton notebook de
    # split. ⚠️ Donne la version "with_spatial_split" (slide COMPLÈTE, buffer inclus),
    # PAS la version "no_buffer" : NOVAE a besoin du voisinage spatial intact.
    inputs: List[str] = field(default_factory=lambda: [
        "/Users/louiselapie/Documents/clip_baseline/Split_ruche/split_screening/cosmx_breast/h5ad/cosmx_breast_rna_with_spatial_split_seed42.h5ad",
        "/Users/louiselapie/Documents/clip_baseline/Split_ruche/split_screening/xenium_renal/h5ad/xenium_renal_rna_with_spatial_split_seed42.h5ad",
    ])
    outdir: str = "novae_embeddings"

    model_name: str = "prism-oncology/novae-human-0"  # alias "MICS-Lab/novae-human-0" accepté aussi

    # Graphe spatial : élague les arêtes plus longues que `radius` (en MICRONS).
    # Mets une valeur adaptée à ton tissu (NOVAE prévient si elle semble mauvaise).
    radius: Optional[float] = 50

    # Facteur de conversion coordonnées -> microns (si coords en pixels).
    # None = coords déjà en microns. CosMx px≈0.1203 ; Xenium px≈0.2125.
    scale_to_microns: Optional[float] = None

    # Colonne obs distinguant les slides si un fichier en contient plusieurs.
    # None = auto-détection (slide_ID / slide_id / Run_Tissue_name / dataset).
    slide_key: Optional[str] = None

    # Colonne obs du split (train/val/test/buffer) à recopier dans le .csv pour
    # filtrer APRÈS le calcul. None = pas de split ; "split" = sortie du notebook.
    split_key: Optional[str] = "split"

    accelerator: str = "cpu"     # "cpu" (défaut, pas de GPU requis), ou "auto"/"cuda"
    num_workers: int = 0         # sur CPU, garder 0 (sinon très lent)
    # CosMx et Xenium sont des tissus indépendants (panels différents) -> on les
    # traite séparément. Les embeddings novae_latent sont identiques de toute façon.
    process_together: bool = False
    save_h5ad: bool = False      # aussi sauver un .h5ad complet (plus lourd)


# ===========================================================================
# SAUVEGARDE (fonction isolée -> testable sans NOVAE ni anndata, voir --selftest)
# ===========================================================================
def save_embeddings(emb: np.ndarray, cell_ids: List[str], slide_ids: List[str],
                    outdir: str, stem: str, splits: Optional[List[str]] = None) -> List[str]:
    """Sauve les embeddings (.npy) + l'index des cellules (.csv) pour réalignement.
    Si `splits` est fourni, ajoute une colonne `split` au .csv (pour filtrer
    train/val/test APRÈS le calcul). Renvoie la liste des chemins écrits."""
    os.makedirs(outdir, exist_ok=True)
    assert emb.ndim == 2, f"emb doit être 2D, reçu {emb.shape}"
    assert emb.shape[0] == len(cell_ids) == len(slide_ids), (
        f"incohérence: {emb.shape[0]} embeddings, {len(cell_ids)} ids, {len(slide_ids)} slides")
    assert splits is None or len(splits) == emb.shape[0], (
        f"incohérence: {emb.shape[0]} embeddings, {len(splits)} splits")

    npy_path = os.path.join(outdir, f"{stem}_novae_latent.npy")
    np.save(npy_path, emb.astype(np.float32))

    header = ["cell_id", "slide_id"] + (["split"] if splits is not None else [])
    csv_path = os.path.join(outdir, f"{stem}_cells.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k, (cid, sid) in enumerate(zip(cell_ids, slide_ids)):
            w.writerow([cid, sid] + ([splits[k]] if splits is not None else []))

    n_zero = int((np.abs(emb).sum(1) == 0).sum())
    print(f"   -> {npy_path}  (shape {emb.shape}, {n_zero} cellule(s) à embedding nul)")
    print(f"   -> {csv_path}" + ("  (avec colonne split)" if splits is not None else ""))
    return [npy_path, csv_path]


# ===========================================================================
# AUTO-DÉTECTION de la colonne 'slide'
# ===========================================================================
def detect_slide_key(adata, user_key: Optional[str]):
    """Renvoie une colonne obs distinguant les slides, ou None si une seule slide."""
    if user_key is not None:
        assert user_key in adata.obs, f"slide_key='{user_key}' absent de adata.obs"
        return user_key
    for cand in ("slide_ID", "slide_id", "Run_Tissue_name", "dataset", "sample"):
        if cand in adata.obs and adata.obs[cand].nunique() > 1:
            print(f"   slide_key auto-détecté: '{cand}' ({adata.obs[cand].nunique()} slides)")
            return cand
    return None


# ===========================================================================
# PIPELINE PRINCIPAL
# ===========================================================================
def run(cfg: Config) -> None:
    import anndata as ad
    import novae

    print("=" * 74)
    print(" NOVAE — précalcul des embeddings ARN (zero-shot, gelé, CPU)")
    print("=" * 74)
    print(f" Modèle      : {cfg.model_name}")
    print(f" Accélérateur: {cfg.accelerator}   (NOVAE ne requiert PAS de GPU)")
    print(f" Sortie      : {cfg.outdir}")
    print("-" * 74)

    if cfg.scale_to_microns is not None:
        novae.settings.scale_to_microns = cfg.scale_to_microns
        print(f" scale_to_microns réglé à {cfg.scale_to_microns}")

    # ---- 1. chargement ----
    adatas, stems, slide_keys = [], [], []
    for path in cfg.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fichier introuvable: {path}\n   Renseigne le bon chemin dans --inputs (ou la config).")
        print(f"\n[chargement] {path}")
        adata = ad.read_h5ad(path)
        print(f"   {adata.n_obs:,} cellules × {adata.n_vars:,} gènes")
        assert "spatial" in adata.obsm, (
            f"adata.obsm['spatial'] absent dans {path}. NOVAE a besoin des coordonnées 2D.")

        # garde-fou comptes bruts : NOVAE veut des comptes bruts (il normalise lui-même)
        xmax = float(adata.X.max())
        if xmax < 10:
            print("   ⚠️  adata.X a un max < 10 : données peut-être déjà normalisées. "
                  "NOVAE attend des COMPTES BRUTS (il fait normalize_total + log1p lui-même).")

        slide_keys.append(detect_slide_key(adata, cfg.slide_key))
        adatas.append(adata)
        stems.append(os.path.splitext(os.path.basename(path))[0])

    # ---- 2. graphe spatial ----
    print("\n[graphe spatial] novae.spatial_neighbors ...")
    for adata, sk in zip(adatas, slide_keys):
        novae.spatial_neighbors(adata, slide_key=sk, radius=cfg.radius)

    # ---- 3. modèle pré-entraîné (gelé) ----
    print(f"\n[modèle] chargement de {cfg.model_name} (téléchargé si 1er usage) ...")
    model = novae.Novae.from_pretrained(cfg.model_name)
    print("   " + repr(model).replace("\n", "\n   "))

    # ---- 4. représentations zero-shot ----
    print("\n[embeddings] compute_representations(zero_shot=True) ...")
    t0 = time.time()
    if cfg.process_together and len(adatas) > 1:
        model.compute_representations(adatas, zero_shot=True,
                                      accelerator=cfg.accelerator, num_workers=cfg.num_workers)
    else:
        for adata in adatas:
            model.compute_representations(adata, zero_shot=True,
                                          accelerator=cfg.accelerator, num_workers=cfg.num_workers)
    print(f"   terminé en {time.time() - t0:.1f}s")

    # ---- 5. sauvegarde ----
    print("\n[sauvegarde]")
    all_paths = []
    for adata, stem, sk in zip(adatas, stems, slide_keys):
        emb = np.asarray(adata.obsm["novae_latent"])
        cell_ids = adata.obs_names.astype(str).tolist()
        if sk is not None and sk in adata.obs:
            slide_ids = adata.obs[sk].astype(str).tolist()
        else:
            slide_ids = [stem] * adata.n_obs
        # colonne split (train/val/test/buffer) -> recopiée dans le .csv pour
        # filtrer APRÈS le calcul (NOVAE a vu la slide entière, voisinages intacts)
        splits = None
        if cfg.split_key and cfg.split_key in adata.obs:
            from collections import Counter
            splits = adata.obs[cfg.split_key].astype(str).tolist()
            print("   split: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(splits).items())))
        all_paths += save_embeddings(emb, cell_ids, slide_ids, cfg.outdir, stem, splits=splits)
        if cfg.save_h5ad:
            h5 = os.path.join(cfg.outdir, f"{stem}_novae.h5ad")
            adata.write_h5ad(h5)
            print(f"   -> {h5}")
            all_paths.append(h5)

    print("\n" + "=" * 74)
    print(f" OK — embeddings ARN sauvegardés dans '{cfg.outdir}'.")
    print(" Réutilise-les directement comme vue ARN du CLIP (alignés par cell_id).")
    print("=" * 74)


# ===========================================================================
# AUTO-TEST de la logique de sauvegarde (sans NOVAE ni anndata)
# ===========================================================================
def selftest() -> bool:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as d:
        emb = np.random.RandomState(0).randn(5, 4)
        emb[2] = 0.0  # une cellule à embedding nul (cas réel: cellule isolée)
        ids = [f"cell{i}" for i in range(5)]
        sids = ["slideA"] * 5
        paths = save_embeddings(emb, ids, sids, d, "test")
        loaded = np.load(paths[0])
        ok &= loaded.shape == (5, 4)
        ok &= np.allclose(loaded, emb.astype(np.float32))
        with open(paths[1]) as f:
            rows = list(csv.reader(f))
        ok &= rows[0] == ["cell_id", "slide_id"]
        ok &= len(rows) == 6 and rows[3] == ["cell2", "slideA"]
        # avec colonne split
        sp = ["train", "train", "buffer", "val", "test"]
        paths2 = save_embeddings(emb, ids, sids, d, "test2", splits=sp)
        with open(paths2[1]) as f:
            rows2 = list(csv.reader(f))
        ok &= rows2[0] == ["cell_id", "slide_id", "split"]
        ok &= rows2[4] == ["cell3", "slideA", "val"]
        # garde-fous de cohérence
        try:
            save_embeddings(emb, ids[:4], sids, d, "bad")
            ok = False  # aurait dû lever (longueurs incohérentes)
        except AssertionError:
            pass
        try:
            save_embeddings(emb, ids, sids, d, "bad2", splits=sp[:3])
            ok = False  # aurait dû lever (splits trop court)
        except AssertionError:
            pass
    print(f" selftest sauvegarde : {'OK ✓' if ok else 'ÉCHEC ✗'}")
    return ok


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    p = argparse.ArgumentParser(description="Précalcul des embeddings NOVAE (zero-shot, local CPU)")
    p.add_argument("--inputs", nargs="+", help="Chemin(s) .h5ad ARN (comptes bruts)")
    p.add_argument("--outdir")
    p.add_argument("--model")
    p.add_argument("--radius", type=float)
    p.add_argument("--scale-microns", type=float, dest="scale_microns")
    p.add_argument("--slide-key", dest="slide_key")
    p.add_argument("--split-key", dest="split_key", help="colonne obs du split (défaut 'split'), '' pour désactiver")
    p.add_argument("--accelerator")
    p.add_argument("--num-workers", type=int, dest="num_workers")
    p.add_argument("--save-h5ad", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    cfg = Config()
    if args.inputs:        cfg.inputs = args.inputs
    if args.outdir:        cfg.outdir = args.outdir
    if args.model:         cfg.model_name = args.model
    if args.radius is not None:        cfg.radius = args.radius
    if args.scale_microns is not None: cfg.scale_to_microns = args.scale_microns
    if args.slide_key:     cfg.slide_key = args.slide_key
    if args.split_key is not None:     cfg.split_key = args.split_key or None
    if args.accelerator:   cfg.accelerator = args.accelerator
    if args.num_workers is not None:   cfg.num_workers = args.num_workers
    if args.save_h5ad:     cfg.save_h5ad = True

    run(cfg)


if __name__ == "__main__":
    main()
