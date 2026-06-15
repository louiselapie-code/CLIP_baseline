"""
Annotation par scConcept (à lancer sur Ruche, GPU + Python >= 3.12).

Pipeline (= conseil du tuteur « annoter avec sc-concept ») :
    1. charge le .h5ad ARN (comptes BRUTS) ;
    2. mappe les symboles de gènes -> Ensembl IDs (helper scConcept) ;
    3. extrait les embeddings cellule scConcept (cls_cell_emb) ;
    4. clustering Leiden SUR ces embeddings  ->  c'est l'annotation ;
    5. sauve labels + embeddings, et un crosstab de contrôle vs l'ancien Leiden.

Note importante : scConcept fournit des EMBEDDINGS, pas des labels. L'« annotation »
est le clustering au-dessus. Pour l'éval (ARI/NMI/sonde) les clusters suffisent ;
on peut nommer les clusters plus tard (par gènes marqueurs) sans changer les métriques.

Installation (Ruche) :  pip install sc-concept scanpy leidenalg igraph
Dépendances : sc-concept (Python>=3.12), scanpy, leidenalg, anndata, numpy, pandas.

La logique de clustering/sortie (étapes 4-5) est testable sans GPU via
    python scconcept_annotate.py --self-test
(elle remplace l'embedding scConcept par un embedding aléatoire sur un petit jeu).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd


# --------------------------------------------------------------------------- #
# Étapes 1-3 : embeddings scConcept (nécessite GPU + Python>=3.12)
# --------------------------------------------------------------------------- #
def scconcept_embed(h5ad_path, model_name, cache_dir, species, counts_layer):
    import scanpy as sc
    from concept import scConcept

    adata = sc.read_h5ad(h5ad_path)
    # comptes BRUTS attendus par le modèle de fondation
    if counts_layer and counts_layer in adata.layers:
        adata.X = adata.layers[counts_layer].copy()
        print(f"  [counts] utilise layers['{counts_layer}']")
    else:
        xmax = adata.X.max()
        print(f"  [counts] utilise X (max={xmax:.1f}) — vérifie que ce sont bien des comptes bruts")

    concept = scConcept(cache_dir=cache_dir)
    concept.load_config_and_model(model_name=model_name)

    adata.var["gene_id"] = concept.map_gene_names_to_ids(
        species=species, gene_names=adata.var_names.tolist()
    )
    mapped = adata.var["gene_id"].notna().sum()
    print(f"  [gènes] {mapped}/{adata.n_vars} symboles mappés vers Ensembl "
          f"({100*mapped/adata.n_vars:.0f}%)")

    result = concept.extract_embeddings(adata=adata, gene_id_column="gene_id")
    emb = np.asarray(result["cls_cell_emb"], dtype=np.float32)
    print(f"  [scConcept] embeddings cellule : {emb.shape}")
    return adata, emb


# --------------------------------------------------------------------------- #
# Étape 4-5 : clustering Leiden + sauvegarde  (TESTABLE sans GPU)
# --------------------------------------------------------------------------- #
def leiden_annotate(emb, resolution, n_neighbors, seed=0):
    import scanpy as sc
    import anndata as ad
    a = ad.AnnData(np.zeros((len(emb), 1), dtype=np.float32))
    a.obsm["X_emb"] = emb.astype(np.float32)
    sc.pp.neighbors(a, use_rep="X_emb", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.leiden(a, resolution=resolution, key_added="lab", random_state=seed,
                 flavor="igraph", n_iterations=2, directed=False)
    return a.obs["lab"].to_numpy().astype(str)


def save_outputs(out_prefix, cell_id, split, labels, emb=None, leiden_ref=None):
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"cell_id": np.asarray(cell_id, dtype=str),
                       "split": np.asarray(split, dtype=str),
                       "scconcept_leiden": np.asarray(labels, dtype=str)})
    df.to_csv(f"{out_prefix}_labels.csv", index=False)
    if emb is not None:
        np.save(f"{out_prefix}_embedding.npy", emb.astype(np.float32))
    print(f"  [save] {out_prefix}_labels.csv  ({df['scconcept_leiden'].nunique()} clusters)")
    print("  distribution:", df["scconcept_leiden"].value_counts().to_dict())
    if leiden_ref is not None:
        ct = pd.crosstab(pd.Series(leiden_ref, name="leiden_ref"), df["scconcept_leiden"])
        ctn = (ct.T / ct.sum(1)).T.round(2)
        print("  crosstab (ancien Leiden -> clusters scConcept) :")
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(ctn.to_string())
    return df


def load_meta(h5ad_path, id_col, split_col):
    import anndata as ad
    a = ad.read_h5ad(h5ad_path, backed="r")
    cid = (a.obs[id_col].astype(str).to_numpy() if id_col and id_col in a.obs
           else np.asarray(a.obs_names, dtype=str))
    split = a.obs[split_col].astype(str).to_numpy() if split_col in a.obs else np.array(["NA"]*a.n_obs)
    ref = a.obs["qc_celltype_cpu"].astype(str).to_numpy() if "qc_celltype_cpu" in a.obs else None
    return cid, split, ref


# --------------------------------------------------------------------------- #
def run(args):
    if args.self_test:
        print("=== SELF-TEST (embedding aléatoire, pas de scConcept) ===")
        n = 3000
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(n, 64)).astype(np.float32)
        emb[:1000] += 4; emb[1000:2000] -= 4   # 3 blobs pour voir des clusters
        labels = leiden_annotate(emb, args.resolution, args.n_neighbors)
        save_outputs(args.out_prefix or "/tmp/selftest", [f"c{i}" for i in range(n)],
                     ["train"]*n, labels, emb=emb,
                     leiden_ref=np.where(np.arange(n)<1000,"A",np.where(np.arange(n)<2000,"B","C")))
        print("SELF-TEST OK")
        return

    adata, emb = scconcept_embed(args.rna_h5ad, args.model, args.cache_dir, args.species, args.counts_layer)
    cid = (adata.obs[args.id_col].astype(str).to_numpy() if args.id_col and args.id_col in adata.obs
           else np.asarray(adata.obs_names, dtype=str))
    split = adata.obs[args.split_col].astype(str).to_numpy() if args.split_col in adata.obs else np.array(["NA"]*adata.n_obs)
    ref = adata.obs["qc_celltype_cpu"].astype(str).to_numpy() if "qc_celltype_cpu" in adata.obs else None
    labels = leiden_annotate(emb, args.resolution, args.n_neighbors)
    save_outputs(args.out_prefix, cid, split, labels, emb=emb, leiden_ref=ref)


def parse():
    p = argparse.ArgumentParser(description="Annotation scConcept (embeddings -> Leiden).")
    p.add_argument("--rna-h5ad", help=".h5ad ARN (comptes bruts, var_names = symboles humains)")
    p.add_argument("--out-prefix", help="préfixe de sortie (…_labels.csv, …_embedding.npy)")
    p.add_argument("--model", default="corpus40M-model30M",
                   help="corpus40M-model30M (humain, défaut) | corpus360M[multi-species]-model170M")
    p.add_argument("--cache-dir", default="./scconcept_cache")
    p.add_argument("--species", default="hsapiens")
    p.add_argument("--counts-layer", default="counts", help="layer de comptes bruts (sinon X)")
    p.add_argument("--id-col", default=None, help="colonne obs cell_id (sinon obs_names)")
    p.add_argument("--split-col", default="split")
    p.add_argument("--resolution", type=float, default=1.0, help="résolution Leiden")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
