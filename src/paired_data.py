"""
Couplage des deux vues du CLIP : ARN (embeddings NOVAE gelés) ↔ protéine (intensités prétraitées).

Les deux vues sont produites séparément :
    ARN     : precompute_novae_embeddings.py  → {stem}_novae_latent.npy   + {stem}_cells.csv
    protéine: preprocess_protein.py           → {stem}_protein_features.npy + {stem}_cells.csv

Chaque .npy a une ligne par cellule, dans le même ordre que son .csv (colonnes cell_id, slide_id, split).
Ce module les apparie **par cell_id** (PAS par ordre de ligne — les deux fichiers peuvent être
ordonnés différemment), et :
  - garde l'intersection des cell_id présents dans les deux vues (inner join) ;
  - **exclut les cellules à embedding ARN nul** (NOVAE renvoie un vecteur 0 pour les cellules à
    comptes nuls — leur z_r serait dégénéré après L2 et polluerait la loss InfoNCE) ;
  - vérifie que le `split` est cohérent entre les deux vues pour chaque cellule.

Renvoie un objet `PairedData` avec des tableaux alignés (rna[i] et protein[i] = même cellule).
Réutilisable tel quel par la future boucle d'entraînement (un Dataset par split).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PairedData:
    rna: np.ndarray        # (M, d_rna)  embeddings NOVAE
    protein: np.ndarray    # (M, d_prot) features protéiques
    cell_id: np.ndarray    # (M,)
    split: np.ndarray      # (M,)  'train' / 'val' / 'test' / 'buffer'

    def __len__(self) -> int:
        return len(self.cell_id)

    def subset(self, split_name: str):
        """Renvoie (rna, protein, cell_id) pour un split donné."""
        m = self.split == split_name
        return self.rna[m], self.protein[m], self.cell_id[m]

    def counts(self) -> dict:
        u, c = np.unique(self.split, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))


def _load_view(npy_path: str, csv_path: str, id_col: str, split_col: str, name: str):
    X = np.load(npy_path)
    df = pd.read_csv(csv_path)
    if len(df) != len(X):
        raise ValueError(f"[{name}] {Path(csv_path).name} ({len(df)} lignes) "
                         f"≠ {Path(npy_path).name} ({len(X)} lignes).")
    if id_col not in df.columns:
        fallback = df.columns[0]
        print(f"[{name}] colonne id '{id_col}' absente → utilise '{fallback}'. "
              f"Colonnes: {list(df.columns)}")
        id_col = fallback
    if split_col not in df.columns:
        raise KeyError(f"[{name}] colonne split '{split_col}' absente. Colonnes: {list(df.columns)}")
    out = pd.DataFrame({
        "cell_id": df[id_col].astype(str).to_numpy(),
        "split": df[split_col].astype(str).to_numpy(),
        "_row": np.arange(len(df)),
    })
    return X, out


def couple(
    rna_npy: str,
    rna_csv: str,
    prot_npy: str,
    prot_csv: str,
    id_col: str = "cell_id",
    split_col: str = "split",
    drop_null_rna: bool = True,
    verbose: bool = True,
) -> PairedData:
    Xr, dr = _load_view(rna_npy, rna_csv, id_col, split_col, "ARN")
    Xp, dp = _load_view(prot_npy, prot_csv, id_col, split_col, "protéine")

    # Doublons de cell_id ? (ne devrait pas arriver, mais on protège le merge)
    for df, nm in [(dr, "ARN"), (dp, "protéine")]:
        d = df["cell_id"].duplicated().sum()
        if d:
            print(f"[warn] {d} cell_id dupliqués côté {nm} — gardera la 1re occurrence.")
    dr = dr.drop_duplicates("cell_id")
    dp = dp.drop_duplicates("cell_id")

    merged = dr.merge(dp, on="cell_id", suffixes=("_rna", "_prot"))
    if len(merged) == 0:
        ex_r = dr["cell_id"].head(3).tolist()
        ex_p = dp["cell_id"].head(3).tolist()
        raise ValueError(
            "Intersection des cell_id VIDE — les identifiants ne correspondent pas entre "
            f"les deux vues.\n  exemples ARN     : {ex_r}\n  exemples protéine: {ex_p}\n"
            "Vérifie le format des cell_id (préfixe slide, indices, etc.)."
        )

    # Cohérence des splits entre vues
    disagree = int((merged["split_rna"] != merged["split_prot"]).sum())
    if disagree:
        print(f"[warn] {disagree} cellules ont un split différent entre ARN et protéine "
              "(on garde celui de l'ARN). À investiguer si > 0.")

    rna = Xr[merged["_row_rna"].to_numpy()]
    protein = Xp[merged["_row_prot"].to_numpy()]
    cell_id = merged["cell_id"].to_numpy()
    split = merged["split_rna"].to_numpy()

    n_inter = len(merged)
    n_null = 0
    if drop_null_rna:
        keep = np.linalg.norm(rna, axis=1) > 0
        n_null = int((~keep).sum())
        rna, protein, cell_id, split = rna[keep], protein[keep], cell_id[keep], split[keep]

    if verbose:
        print("Couplage ARN↔protéine :")
        print(f"  ARN      : {len(dr):>8d} cellules ({Xr.shape[1]} dims)")
        print(f"  protéine : {len(dp):>8d} cellules ({Xp.shape[1]} dims)")
        print(f"  intersection cell_id : {n_inter}")
        if drop_null_rna:
            print(f"  exclues (embedding ARN nul) : {n_null}")
        print(f"  → paires finales : {len(cell_id)}")
        print(f"  répartition split : {dict(sorted({s:int((split==s).sum()) for s in set(split)}.items()))}")

    return PairedData(rna=rna, protein=protein, cell_id=cell_id, split=split)


def save_paired(data: PairedData, outdir: str) -> None:
    """Sauvegarde les vues alignées (entrée prête pour la boucle d'entraînement)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "paired_rna.npy", data.rna.astype(np.float32))
    np.save(out / "paired_protein.npy", data.protein.astype(np.float32))
    pd.DataFrame({"cell_id": data.cell_id, "split": data.split}).to_csv(
        out / "paired_cells.csv", index=False
    )
    print(f"  sauvé : paired_rna.npy {data.rna.shape}, paired_protein.npy {data.protein.shape}, "
          f"paired_cells.csv dans {out}/")
