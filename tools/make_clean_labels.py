import anndata as ad

# CosMx : on EXCLUT les cellules-artefacts en les OMETTANT (filtre par cellule, pas par type)
a = ad.read_h5ad("annotation/cosmx_annotated.h5ad")              # adapte le chemin
NCOUNT_MIN = 150                                       # décrochage réel : B 11 / T 19 / NormEpi 93 / Plasma 252 / autres 785+
keep = a.obs["nCount_RNA"] >= NCOUNT_MIN
(a.obs.loc[keep, ["cell_id", "cell_type_pred"]]
   .rename(columns={"cell_type_pred": "cell_type"})
   .to_csv("annotation/labels/cosmx_labels.csv", index=False))
print(f"CosMx: {int(keep.sum())}/{a.n_obs} cellules, types: {sorted(a.obs.loc[keep,'cell_type_pred'].unique())}")

# Xenium : annotation déjà propre, pas de filtrage
x = ad.read_h5ad("annotation/xenium_annotated.h5ad")             # adapte le chemin
(x.obs[["cell_id", "cell_type_pred"]]
   .rename(columns={"cell_type_pred": "cell_type"})
   .to_csv("annotation/labels/xenium_labels.csv", index=False))
print(f"Xenium: {x.n_obs} cellules, types: {sorted(x.obs['cell_type_pred'].unique())}")