"""
Encodeur protéique + tête de projection — côté protéine du modèle CLIP ARN–protéine.

Référence : « Stratégie d'entraînement », §2.3 (encodeur MLP shallow) et §2.4 (tête de projection).

Chaîne côté protéine :

    p  (R^P, intensités d'anticorps prétraitées par preprocess_protein.py)
      │
      ▼  ProteinEncoder  (MLP shallow : Linear → LayerNorm → GELU → Dropout, ×depth)
    z_latent  (R^{latent_dim}, 128 par défaut)
      │
      ▼  ProteinProjectionHead  (Linear sans activation → normalisation L2)
    z_p  (R^{dproj}, 256 par défaut, ||z_p|| = 1)

Ce sont les SEULS modules entraînés côté protéine (l'encodeur ARN NOVAE reste gelé).
On les regroupe dans `ProteinTower` pour les brancher facilement dans le modèle CLIP.

Hyperparamètres par défaut alignés sur le tableau §5.1 :
    hidden_dim=128, latent_dim=128, dproj=256, depth=2, dropout=0.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ProteinTowerConfig:
    """Hyperparamètres du côté protéine (cf. §5.1)."""
    in_dim: int                 # P = nombre de marqueurs APRÈS exclusion des canaux techniques
    hidden_dim: int = 128       # dimension cachée du MLP
    latent_dim: int = 128       # dimension de z_latent (sortie de l'encodeur)
    dproj: int = 256            # dimension de l'espace contrastif commun
    depth: int = 2              # nombre de couches linéaires de l'encodeur (∈ {1, 2, 3})
    dropout: float = 0.1        # dropout du MLP protéique
    proj_bias: bool = False     # biais de la tête de projection (CLIP standard : sans biais)


# --------------------------------------------------------------------------- #
# Encodeur protéique : MLP peu profond  (§2.3)
# --------------------------------------------------------------------------- #
class ProteinEncoder(nn.Module):
    """MLP shallow : R^{in_dim} → R^{latent_dim}.

    Bloc élémentaire répété `depth` fois : Linear → LayerNorm → GELU → Dropout.
    Le MLP reste volontairement simple (in_dim faible) pour limiter l'overfitting.

    depth=1 : Linear(in_dim, latent_dim)
    depth=2 : Linear(in_dim, hidden_dim) → Linear(hidden_dim, latent_dim)   (défaut)
    depth=3 : Linear(in_dim, hidden) → Linear(hidden, hidden) → Linear(hidden, latent_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth doit être >= 1 (reçu : {depth})")
        if in_dim < 1:
            raise ValueError(f"in_dim doit être >= 1 (reçu : {in_dim})")

        # Dimensions successives des couches linéaires.
        dims = [in_dim] + [hidden_dim] * (depth - 1) + [latent_dim]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, in_dim) → z_latent : (N, latent_dim)."""
        return self.net(x)


# --------------------------------------------------------------------------- #
# Tête de projection protéique  (§2.4)
# --------------------------------------------------------------------------- #
class ProteinProjectionHead(nn.Module):
    """Linear(latent_dim, dproj) SANS activation, suivie d'une normalisation L2.

    La normalisation L2 garantit ||z_p|| = 1, de sorte que le produit scalaire
    z_r·z_p utilisé dans la loss InfoNCE soit exactement la similarité cosinus.
    """

    def __init__(self, latent_dim: int = 128, dproj: int = 256, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dproj, bias=bias)
        self.latent_dim = latent_dim
        self.dproj = dproj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, latent_dim) → z_p : (N, dproj), normalisé L2."""
        x = self.proj(x)
        return F.normalize(x, p=2, dim=-1)


# --------------------------------------------------------------------------- #
# Tour protéique complète : encodeur + tête de projection
# --------------------------------------------------------------------------- #
class ProteinTower(nn.Module):
    """Encodeur protéique + tête de projection : R^P → z_p (R^{dproj}, normalisé L2).

    À utiliser côté protéine du modèle CLIP. L'équivalent côté ARN est :
        embedding NOVAE (gelé, précalculé) → tête de projection ARN Linear(64, 256) + L2.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        dproj: int = 256,
        depth: int = 2,
        dropout: float = 0.1,
        proj_bias: bool = False,
    ):
        super().__init__()
        self.encoder = ProteinEncoder(in_dim, hidden_dim, latent_dim, depth, dropout)
        self.head = ProteinProjectionHead(latent_dim, dproj, bias=proj_bias)
        self.in_dim = in_dim
        self.dproj = dproj

    @classmethod
    def from_config(cls, cfg: ProteinTowerConfig) -> "ProteinTower":
        return cls(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            dproj=cfg.dproj,
            depth=cfg.depth,
            dropout=cfg.dropout,
            proj_bias=cfg.proj_bias,
        )

    def forward(
        self, x: torch.Tensor, return_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """x : (N, in_dim) → z_p : (N, dproj).

        Si return_latent=True, renvoie aussi z_latent (utile pour la sonde linéaire / debug).
        """
        z_latent = self.encoder(x)
        z_p = self.head(z_latent)
        if return_latent:
            return z_p, z_latent
        return z_p


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def init_weights(module: nn.Module, seed: int | None = None) -> None:
    """Initialisation reproductible (Xavier uniforme pour les Linear, cf. §4 « fixer une seed »).

    Appeler `model.apply(lambda m: init_weights(m))` après avoir fixé la seed globale,
    ou utiliser `set_seed(...)` ci-dessous avant de construire le modèle.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def set_seed(seed: int) -> None:
    """Fixe les seeds (Python hash exclu) pour des runs reproductibles."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not only_trainable)


# --------------------------------------------------------------------------- #
# Smoke test : python protein_encoder.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    set_seed(42)
    P, N = 60, 8  # 60 marqueurs, batch de 8 cellules

    tower = ProteinTower(in_dim=P, depth=2, dropout=0.1)
    tower.apply(lambda m: init_weights(m))

    x = torch.randn(N, P)
    z_p, z_latent = tower(x, return_latent=True)

    norms = z_p.norm(dim=-1)
    print(f"ProteinTower : in_dim={P}, latent={tower.encoder.latent_dim}, dproj={tower.dproj}")
    print(f"  entrée            : {tuple(x.shape)}")
    print(f"  z_latent          : {tuple(z_latent.shape)}")
    print(f"  z_p (projeté)     : {tuple(z_p.shape)}")
    print(f"  ||z_p|| (≈1)      : min={norms.min():.4f}  max={norms.max():.4f}")
    print(f"  paramètres entraînables : {count_parameters(tower):,}")

    assert z_p.shape == (N, tower.dproj)
    assert torch.allclose(norms, torch.ones(N), atol=1e-5), "z_p doit être normalisé L2"
    print("OK ✓")
