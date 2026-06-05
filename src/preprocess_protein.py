"""
Préprocessing des données protéiques CosMx — côté protéine du modèle CLIP ARN–protéine.

Référence : « Stratégie d'entraînement », §2.3 (« Pré-traitement des données protéiques »).

Les données protéiques sont des intensités d'anticorps (comme CODEX/IMC). Pipeline :

    1. Exclusion des canaux techniques        (liste explicite fournie par l'utilisateur)
    2. Clipping des valeurs extrêmes           (percentile 1–99 par marqueur)
    3. Transformation log(1 + x)
    4. Standardisation par marqueur            (mean=0, std=1)

POINT DE CORRECTION IMPORTANT (non précisé dans le doc, mais nécessaire) :
    les statistiques de normalisation (bornes de clipping, mean, std) sont AJUSTÉES SUR
    LE SPLIT TRAIN UNIQUEMENT, puis appliquées à val/test. Sinon il y a fuite de données
    (les stats de test « voient » la distribution de test) et l'évaluation est faussée.
    Les stats sont sauvegardées en JSON pour être réappliquées à l'identique à l'inférence.

Entrée  : un .h5ad protéique séparé par slide.
            adata.X            = intensités brutes (cellules × marqueurs)
            adata.var_names    = noms des marqueurs (canaux)
            adata.obs['split'] = 'train' / 'val' / 'test' / 'buffer'  (cf. côté ARN)
            adata.obs_names    = cell_id (alignable avec la vue ARN par cell_id)

Sorties (par fichier, dans --outdir) :
    {stem}_protein_features.npy      matrice (N, P) float32, prête pour ProteinTower
    {stem}_cells.csv                 cell_id, slide_id, split  (même ordre que le .npy)
    {stem}_protein_norm_stats.json   marqueurs gardés + bornes clip + mean/std (train)

Le schéma de {stem}_cells.csv est identique à celui du précalcul ARN
(precompute_novae_embeddings.py) → réalignement ARN↔protéine par cell_id.

Usage :
    python preprocess_protein.py \
        --inputs cosmx_protein_with_spatial_split_seed42.h5ad \
        --outdir ./protein_features \
        --exclude IgG1 IgG2a IgG2b Negative DAPI        # <-- liste exacte à fournir
        # ou : --exclude-file canaux_techniques.txt     (un nom par ligne)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    inputs: list[str]
    outdir: str = "./protein_features"
    exclude: list[str] = field(default_factory=list)   # canaux techniques (noms EXACTS)
    split_col: str = "split"
    slide_col: str = "slide_id"      # colonne obs pour le slide_id (sinon : stem du fichier)
    id_col: str | None = None        # colonne obs pour cell_id (sinon : adata.obs_names)
    clip_low: float = 1.0            # percentile bas du clipping
    clip_high: float = 99.0          # percentile haut du clipping
    log1p: bool = True               # appliquer log(1+x) (désactiver si données déjà transformées)
    fit_on: str = "train"            # ajuster les stats sur {"train", "all"}
    std_eps: float = 1e-8            # plancher pour éviter une division par std≈0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_dense(X) -> np.ndarray:
    """Densifie une matrice (sparse ou non) en float64 pour les calculs de stats."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def select_markers(
    marker_names: list[str], exclude: list[str]
) -> tuple[list[int], list[str]]:
    """Retourne (indices gardés, noms gardés) après exclusion des canaux techniques (noms exacts)."""
    exclude_set = set(exclude)
    not_found = exclude_set - set(marker_names)
    if not_found:
        print(f"  [warn] canaux à exclure absents du panel (ignorés) : {sorted(not_found)}")
    keep_idx = [i for i, m in enumerate(marker_names) if m not in exclude_set]
    keep_names = [marker_names[i] for i in keep_idx]
    if not exclude:
        print("  [warn] aucune liste d'exclusion fournie (--exclude / --exclude-file) : "
              "TOUS les canaux sont gardés, y compris d'éventuels canaux techniques.")
    return keep_idx, keep_names


def fit_norm_stats(
    X_train: np.ndarray,
    marker_names: list[str],
    clip_low: float,
    clip_high: float,
    log1p: bool,
    std_eps: float,
) -> dict:
    """Ajuste les statistiques de normalisation SUR LE TRAIN uniquement.

    Ordre (cf. §2.3) : clip(p1, p99) → log1p → standardisation (mean/std).
    Les bornes de clip sont calculées sur les intensités brutes du train ;
    mean/std sont calculés sur les valeurs APRÈS clip + log1p du train.
    """
    lo = np.percentile(X_train, clip_low, axis=0)
    hi = np.percentile(X_train, clip_high, axis=0)
    # Garantit lo <= hi même pour un marqueur quasi constant.
    hi = np.maximum(hi, lo)

    clipped = np.clip(X_train, lo, hi)
    if log1p:
        if np.nanmin(clipped) < 0:
            print("  [warn] valeurs négatives détectées avant log1p (données déjà transformées ?). "
                  "Elles sont ramenées à 0. Utilise --no-log1p si tes données ne sont pas des comptes bruts.")
            clipped = np.maximum(clipped, 0.0)
        transformed = np.log1p(clipped)
    else:
        transformed = clipped

    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)
    std = np.where(std < std_eps, 1.0, std)  # marqueur constant → on ne divise pas (std=1)

    return {
        "markers": list(marker_names),
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "log1p": bool(log1p),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_fit": int(X_train.shape[0]),
    }


def apply_norm(X: np.ndarray, stats: dict) -> np.ndarray:
    """Applique les stats (ajustées sur train) à n'importe quel sous-ensemble de cellules."""
    lo = np.asarray(stats["lo"])
    hi = np.asarray(stats["hi"])
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])

    clipped = np.clip(X, lo, hi)
    if stats["log1p"]:
        clipped = np.maximum(clipped, 0.0)
        transformed = np.log1p(clipped)
    else:
        transformed = clipped
    return ((transformed - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------- #
# Traitement d'un fichier
# --------------------------------------------------------------------------- #
def process_file(path: str, cfg: Config, outdir: Path) -> None:
    import anndata as ad  # import tardif : permet d'utiliser les fonctions sans anndata

    stem = Path(path).stem
    print(f"\n=== {stem} ===")
    adata = ad.read_h5ad(path)
    n_cells = adata.n_obs
    marker_names = list(map(str, adata.var_names))
    print(f"  {n_cells} cellules × {len(marker_names)} canaux")

    # 1) Exclusion des canaux techniques
    keep_idx, keep_names = select_markers(marker_names, cfg.exclude)
    X = _to_dense(adata.X)[:, keep_idx]
    print(f"  {len(keep_names)} marqueurs gardés après exclusion (P = {len(keep_names)})")

    # Métadonnées cellules
    if cfg.id_col and cfg.id_col in adata.obs:
        cell_id = adata.obs[cfg.id_col].astype(str).to_numpy()
    else:
        cell_id = np.asarray(adata.obs_names, dtype=str)
    if cfg.slide_col in adata.obs:
        slide_id = adata.obs[cfg.slide_col].astype(str).to_numpy()
    else:
        print(f"  [warn] colonne slide '{cfg.slide_col}' absente : slide_id = '{stem}'")
        slide_id = np.full(n_cells, stem, dtype=object)
    if cfg.split_col not in adata.obs:
        raise KeyError(
            f"Colonne split '{cfg.split_col}' absente de obs. "
            f"Colonnes disponibles : {list(adata.obs.columns)}"
        )
    split = adata.obs[cfg.split_col].astype(str).to_numpy()

    # 2-4) Ajustement des stats SUR TRAIN, puis application à toutes les cellules
    if cfg.fit_on == "train":
        fit_mask = split == "train"
        if fit_mask.sum() == 0:
            raise ValueError(
                f"Aucune cellule 'train' dans '{cfg.split_col}' "
                f"(valeurs vues : {sorted(set(split))}). Utilise --fit-on all si volontaire."
            )
    else:  # "all"
        print("  [warn] --fit-on all : stats ajustées sur TOUTES les cellules (risque de fuite).")
        fit_mask = np.ones(n_cells, dtype=bool)

    stats = fit_norm_stats(
        X[fit_mask], keep_names, cfg.clip_low, cfg.clip_high, cfg.log1p, cfg.std_eps
    )
    print(f"  stats ajustées sur {stats['n_fit']} cellules ({cfg.fit_on})")
    features = apply_norm(X, stats)  # (N, P) float32

    # Sorties (ordre des lignes == ordre des cellules du .h5ad, comme le côté ARN)
    npy_path = outdir / f"{stem}_protein_features.npy"
    csv_path = outdir / f"{stem}_cells.csv"
    json_path = outdir / f"{stem}_protein_norm_stats.json"

    np.save(npy_path, features)
    _write_cells_csv(csv_path, cell_id, slide_id, split)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Contrôle rapide : sur le TRAIN, après normalisation, mean≈0 et std≈1.
    tr = features[fit_mask] if cfg.fit_on == "train" else features
    print(f"  contrôle train  → mean={tr.mean():+.3f}  std={tr.std():.3f}  "
          f"(attendu ≈ 0 / ≈ 1)")
    counts = {s: int((split == s).sum()) for s in sorted(set(split))}
    print(f"  répartition split : {counts}")
    print(f"  écrit : {npy_path.name}, {csv_path.name}, {json_path.name}")


def _write_cells_csv(path: Path, cell_id, slide_id, split) -> None:
    """Écrit cell_id, slide_id, split (sans dépendre de pandas)."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "slide_id", "split"])
        for c, s, sp in zip(cell_id, slide_id, split):
            w.writerow([c, s, sp])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Préprocessing protéique pour le CLIP ARN–protéine.")
    p.add_argument("--inputs", nargs="+", required=True, help="Fichiers .h5ad protéiques (un par slide).")
    p.add_argument("--outdir", default="./protein_features")
    p.add_argument("--exclude", nargs="*", default=[], help="Canaux techniques à exclure (noms EXACTS).")
    p.add_argument("--exclude-file", default=None, help="Fichier texte : un nom de canal par ligne.")
    p.add_argument("--split-col", default="split")
    p.add_argument("--slide-col", default="slide_id")
    p.add_argument("--id-col", default=None, help="Colonne obs pour cell_id (défaut : obs_names).")
    p.add_argument("--clip-low", type=float, default=1.0)
    p.add_argument("--clip-high", type=float, default=99.0)
    p.add_argument("--no-log1p", action="store_true", help="Désactive log(1+x).")
    p.add_argument("--fit-on", choices=["train", "all"], default="train")
    a = p.parse_args()

    exclude = list(a.exclude)
    if a.exclude_file:
        with open(a.exclude_file) as f:
            exclude += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    return Config(
        inputs=a.inputs,
        outdir=a.outdir,
        exclude=exclude,
        split_col=a.split_col,
        slide_col=a.slide_col,
        id_col=a.id_col,
        clip_low=a.clip_low,
        clip_high=a.clip_high,
        log1p=not a.no_log1p,
        fit_on=a.fit_on,
    )


def main() -> None:
    cfg = parse_args()
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("Configuration :")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    for path in cfg.inputs:
        process_file(path, cfg, outdir)
    print("\nTerminé ✓")


if __name__ == "__main__":
    main()
