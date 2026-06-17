"""
niches.py — « fin de NOVAE » appliquée à des embeddings cellulaires arbitraires.

Reproduit fidèlement la partie INFÉRENCE (zero-shot) de NOVAE, pour transformer des
embeddings cellulaires (ici la sortie de ton CLIP, ou NOVAE/scConcept bruts) en niches
spatiales, SANS ré-entraîner NOVAE :

    1. projection sur la sphère unité (L2)
    2. prototypes par KMeans sur les embeddings L2-normalisés        (K prototypes)
    3. attribution d'un prototype « feuille » à chaque cellule       (argmax cosinus)
    4. clustering hiérarchique des prototypes (cosine, average)      → arbre de domaines
    5. mapping feuille → domaine (niche) à un niveau / n_domains donné

Correspondance avec le code NOVAE (MICS-Lab/novae) :
    - étape 1/2 : novae/module/swav.py  ::  SwavHead.compute_kmeans_prototypes
    - étape 3   : novae/model.py        ::  Novae._compute_leaves           (argmax cos)
    - étape 4   : novae/module/swav.py  ::  SwavHead.hierarchical_clustering
    - étape 5   : novae/module/swav.py  ::  SwavHead.map_leaves_domains / find_level
    - variante Sinkhorn : novae/module/swav.py :: SwavHead.sinkhorn

⚠️ Nuance importante : dans NOVAE, le transport optimal (Sinkhorn-Knopp) sert pendant
   l'ENTRAÎNEMENT (cible de la loss SwAV). À l'inférence, NOVAE attribue les prototypes
   par simple argmax de similarité cosinus. On reproduit l'inférence par défaut
   (`assign="argmax"`) ; `assign="sinkhorn"` est fourni comme variante OT au moment de
   l'attribution (plus proche de la description « transport optimal » mais NON utilisé
   par NOVAE à l'inférence).

Dépendances : numpy + scikit-learn. Leiden (optionnel) : scanpy. Lissage : scipy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Constantes NOVAE (novae/_constants.py :: Nums)
EPS: float = 1e-8
SWAV_EPSILON: float = 0.05
SINKHORN_ITERATIONS: int = 3
DEFAULT_NUM_PROTOTYPES: int = 512
DEFAULT_SAMPLE_CELLS: int = 100_000


# --------------------------------------------------------------------------- #
# Étape 1-2 : sphère + prototypes
# --------------------------------------------------------------------------- #
def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Projection sur la sphère unité (norme L2 = 1 par ligne)."""
    x = np.asarray(x, dtype=np.float64)
    return x / (EPS + np.linalg.norm(x, axis=axis, keepdims=True))


def compute_kmeans_prototypes(
    latent: np.ndarray,
    num_prototypes: int = DEFAULT_NUM_PROTOTYPES,
    sample_cells: int | None = DEFAULT_SAMPLE_CELLS,
    seed: int = 0,
) -> np.ndarray:
    """Prototypes = centres KMeans sur les embeddings L2-normalisés, eux-mêmes renormalisés.

    Port de `SwavHead.compute_kmeans_prototypes` (+ sous-échantillonnage de
    `Novae.init_prototypes`, qui ajuste les prototypes sur <= DEFAULT_SAMPLE_CELLS cellules).

    Args:
        latent: embeddings (N, D).
        num_prototypes: nombre de prototypes K (NOVAE: 512).
        sample_cells: si N > sample_cells, ajuste le KMeans sur un sous-échantillon (vitesse).
        seed: graine.

    Returns:
        prototypes (K, D), L2-normalisés.
    """
    from sklearn.cluster import KMeans

    X = l2_normalize(np.asarray(latent, dtype=np.float64))
    if sample_cells and len(X) > sample_cells:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), sample_cells, replace=False)]
    assert len(X) >= num_prototypes, (
        f"#cellules ({len(X)}) < #prototypes ({num_prototypes}). Réduis num_prototypes."
    )
    km = KMeans(n_clusters=num_prototypes, random_state=seed, n_init="auto").fit(X)
    proto = km.cluster_centers_
    proto = proto / (EPS + np.linalg.norm(proto, axis=1, keepdims=True))
    return proto.astype(np.float32)


# --------------------------------------------------------------------------- #
# Étape 3 : attribution des feuilles (prototypes)
# --------------------------------------------------------------------------- #
def projection(z: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Similarité cosinus entre embeddings (normalisés) et prototypes (déjà normalisés).

    Port de `SwavHead.projection`. Retourne (N, K).
    """
    return (l2_normalize(np.asarray(z, dtype=np.float64)) @ np.asarray(prototypes, dtype=np.float64).T)


def sinkhorn(projections: np.ndarray, epsilon: float = SWAV_EPSILON, n_iters: int = SINKHORN_ITERATIONS) -> np.ndarray:
    """Sinkhorn-Knopp : code doux équilibré (transport optimal). Port exact de `SwavHead.sinkhorn`.

    NB : appliqué ici sur l'ensemble des cellules à la fois (OT global), au lieu d'un
    mini-batch comme à l'entraînement. Mémoire O(N*K).
    """
    Q = np.exp(np.asarray(projections, dtype=np.float64) / epsilon)
    Q /= Q.sum()
    B, K = Q.shape
    for _ in range(n_iters):
        Q /= Q.sum(axis=0, keepdims=True)
        Q /= K
        Q /= Q.sum(axis=1, keepdims=True)
        Q /= B
    return Q / Q.sum(axis=1, keepdims=True)  # lignes -> 1


def assign_leaves(z: np.ndarray, prototypes: np.ndarray, method: str = "argmax") -> np.ndarray:
    """Attribue à chaque cellule l'indice de son prototype « feuille ».

    method="argmax"   : argmax de la similarité cosinus (= inférence NOVAE).
    method="sinkhorn" : argmax du code Sinkhorn-OT (variante).
    """
    P = projection(z, prototypes)  # (N, K)
    if method == "sinkhorn":
        P = sinkhorn(P)
    elif method != "argmax":
        raise ValueError(f"method inconnu : {method!r} (argmax|sinkhorn)")
    return P.argmax(axis=1).astype(np.int64)


# --------------------------------------------------------------------------- #
# Étape 4 : clustering hiérarchique des prototypes
# --------------------------------------------------------------------------- #
def hierarchical_clustering(prototypes: np.ndarray):
    """Clustering agglomératif (cosine, average) des prototypes → arbre complet.

    Port de `SwavHead.hierarchical_clustering`. Construit la matrice `clusters_levels`
    de forme (K, K) : la ligne `r` contient l'étiquette de domaine de chaque prototype
    quand il reste K-r domaines (ligne 0 = K singletons, dernière ligne = 1 domaine).

    Returns:
        (clustering sklearn, clusters_levels (K, K)).
    """
    from sklearn.cluster import AgglomerativeClustering

    X = np.asarray(prototypes, dtype=np.float64)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        compute_full_tree=True,
        metric="cosine",
        linkage="average",
    ).fit(X)

    K = len(X)
    clusters_levels = np.zeros((K, K), dtype=np.int64)
    clusters_levels[0] = np.arange(K)
    for i, (a, b) in enumerate(clustering.children_):
        clusters = clusters_levels[i]
        clusters_levels[i + 1] = clusters
        clusters_levels[i + 1, np.where((clusters == a) | (clusters == b))] = K + i
    return clustering, clusters_levels


# --------------------------------------------------------------------------- #
# Étape 5 : feuille → domaine (niche)
# --------------------------------------------------------------------------- #
def leaves_to_domains(leaves: np.ndarray, clusters_levels: np.ndarray, level: int) -> np.ndarray:
    """Mappe chaque feuille (indice de prototype) vers son ancêtre au `level` donné.

    Port de `SwavHead.map_leaves_domains`. `clusters_levels[-level]` donne exactement
    `level` domaines distincts (cf. construction de l'arbre).
    """
    mapping = clusters_levels[-level]  # (K,) : prototype -> id de domaine brut
    return mapping[np.asarray(leaves, dtype=np.int64)]


def find_level(clusters_levels: np.ndarray, leaves_present: np.ndarray, n_domains: int) -> int:
    """Trouve le niveau de coupe donnant exactement `n_domains` domaines sur les feuilles présentes.

    Port de `SwavHead.find_level`.
    """
    K = clusters_levels.shape[1]
    sub = clusters_levels[:, np.asarray(leaves_present, dtype=np.int64)]
    for level in range(1, K):
        if len(np.unique(sub[-level])) == n_domains:
            return level
    raise ValueError(f"Aucun niveau ne donne {n_domains} domaines (essaie une autre valeur).")


def leiden_prototypes(prototypes: np.ndarray, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Clustering Leiden des prototypes (alternative au hiérarchique, recommandée par NOVAE en zero-shot).

    Port de `Novae._leiden_prototypes`. Nécessite scanpy. Retourne un code (K,) prototype -> cluster.
    """
    import anndata as ad
    import scanpy as sc

    ap = ad.AnnData(np.asarray(prototypes, dtype=np.float32))
    sc.pp.pca(ap)
    sc.pp.neighbors(ap)
    sc.tl.leiden(ap, flavor="igraph", resolution=resolution, random_state=seed, n_iterations=2, directed=False)
    return ap.obs["leiden"].values.codes.astype(np.int64)


# --------------------------------------------------------------------------- #
# Lissage spatial optionnel (pour renforcer le signal « niche »)
# --------------------------------------------------------------------------- #
def spatial_smooth(embeddings: np.ndarray, coords: np.ndarray, k: int = 10) -> np.ndarray:
    """Moyenne l'embedding de chaque cellule sur ses k plus proches voisins spatiaux (soi inclus).

    Utile si l'embedding est purement cellulaire (ex. côté protéine) : sans contexte
    spatial, clusteriser donne des types cellulaires plutôt que des niches.
    """
    from scipy.spatial import cKDTree

    Z = np.asarray(embeddings, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float64)
    tree = cKDTree(coords)
    _, nn = tree.query(coords, k=k + 1)  # inclut la cellule elle-même
    if nn.ndim == 1:
        nn = nn[:, None]
    return Z[nn].mean(axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Interface haut-niveau
# --------------------------------------------------------------------------- #
@dataclass
class NicheResult:
    leaves: np.ndarray  # (N,) indice de prototype par cellule
    domains: np.ndarray  # (N,) id de niche (0..n_domains-1) par cellule
    prototypes: np.ndarray  # (K, D)
    clusters_levels: np.ndarray  # (K, K)
    level: int  # niveau de coupe utilisé
    n_domains: int  # nb de domaines obtenus


def assign_niches(
    embeddings: np.ndarray,
    *,
    num_prototypes: int = DEFAULT_NUM_PROTOTYPES,
    n_domains: int | None = 10,
    level: int | None = None,
    niche_method: str = "hierarchical",
    resolution: float = 1.0,
    assign: str = "argmax",
    prototypes: np.ndarray | None = None,
    sample_cells: int | None = DEFAULT_SAMPLE_CELLS,
    seed: int = 0,
) -> NicheResult:
    """Pipeline complet « fin de NOVAE » : embeddings (N, D) → niches (N,).

    Args:
        embeddings: matrice (N, D) d'embeddings cellulaires.
        num_prototypes: nombre de prototypes K (KMeans).
        n_domains: nombre de niches voulu (hiérarchique). Ignoré si `level` est fourni.
        level: niveau de coupe direct (sinon déterminé via `n_domains`).
        niche_method: "hierarchical" (défaut, coupe d'arbre) ou "leiden" (sur les prototypes).
        resolution: résolution Leiden (si niche_method="leiden").
        assign: "argmax" (inférence NOVAE) ou "sinkhorn" (variante OT).
        prototypes: prototypes pré-calculés (K, D) pour réutiliser un référentiel ; sinon KMeans.
        sample_cells: sous-échantillon pour ajuster le KMeans.
        seed: graine.

    Returns:
        NicheResult.
    """
    Z = np.asarray(embeddings, dtype=np.float32)
    if prototypes is None:
        prototypes = compute_kmeans_prototypes(Z, num_prototypes, sample_cells, seed)
    prototypes = np.asarray(prototypes, dtype=np.float32)

    leaves = assign_leaves(Z, prototypes, method=assign)

    if niche_method == "leiden":
        codes = leiden_prototypes(prototypes, resolution=resolution, seed=seed)
        raw_domains = codes[leaves]
        used_level = -1
    elif niche_method == "hierarchical":
        _, clusters_levels = hierarchical_clustering(prototypes)
        if level is None:
            assert n_domains is not None, "Fournis `n_domains` ou `level`."
            used_level = find_level(clusters_levels, np.unique(leaves), n_domains)
        else:
            used_level = level
        raw_domains = leaves_to_domains(leaves, clusters_levels, used_level)
    else:
        raise ValueError(f"niche_method inconnu : {niche_method!r} (hierarchical|leiden)")

    # Relabel contigu 0..n-1
    uniq = {d: i for i, d in enumerate(np.unique(raw_domains))}
    domains = np.array([uniq[d] for d in raw_domains], dtype=np.int64)

    if niche_method != "hierarchical":
        clusters_levels = np.zeros((len(prototypes), len(prototypes)), dtype=np.int64)

    return NicheResult(
        leaves=leaves,
        domains=domains,
        prototypes=prototypes,
        clusters_levels=clusters_levels,
        level=used_level,
        n_domains=len(uniq),
    )


# --------------------------------------------------------------------------- #
# Smoke test : python niches.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 4 blobs nettement séparés -> doit retrouver ~4 domaines
    centers = rng.normal(size=(4, 16))
    Z = np.concatenate([c + 0.05 * rng.normal(size=(500, 16)) for c in centers])
    res = assign_niches(Z, num_prototypes=32, n_domains=4, sample_cells=None, seed=0)
    print("leaves uniques :", len(np.unique(res.leaves)), "| niveau :", res.level)
    print("domaines :", res.n_domains, "| tailles :", np.bincount(res.domains))
    # vérif : argmax cos == plus proche prototype
    P = projection(Z, res.prototypes)
    assert (P.argmax(1) == res.leaves).all()
    # vérif Sinkhorn : lignes somment à 1
    q = sinkhorn(P[:100])
    assert np.allclose(q.sum(1), 1.0, atol=1e-6)
    print("smoke test OK ✓")
