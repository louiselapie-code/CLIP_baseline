"""
Modèle CLIP ARN–protéine + loss InfoNCE symétrique (§2.1, §2.4).

Assemble les deux tours dans l'espace contrastif commun (dim 256) :

    ARN  : embedding NOVAE (gelé, précalculé, dim 64)
             → RNAProjectionHead : Linear(64, 256) sans activation + L2   → z_r
    prot : intensités prétraitées (dim P)
             → ProteinTower (MLP + Linear(128,256) + L2)                  → z_p

Loss = InfoNCE symétrique (CLIP), température τ fixée à 0.07.
Tous les paramètres du modèle sont entraînés (NOVAE n'est PAS dans le modèle : ses
embeddings sont une entrée figée). On suit séparément les deux directions de la loss
pour détecter un déséquilibre entre modalités (§6.2).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_encoder import ProteinTower


# --------------------------------------------------------------------------- #
# Tête de projection ARN
# --------------------------------------------------------------------------- #
class RNAProjectionHead(nn.Module):
    """Linear(rna_dim, dproj) sans activation, suivie d'une normalisation L2."""

    def __init__(self, rna_dim: int = 64, dproj: int = 256, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(rna_dim, dproj, bias=bias)
        self.rna_dim = rna_dim
        self.dproj = dproj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=-1)


# --------------------------------------------------------------------------- #
# Modèle CLIP
# --------------------------------------------------------------------------- #
@dataclass
class CLIPConfig:
    rna_dim: int = 64
    prot_dim: int = 64
    dproj: int = 256
    depth: int = 2
    hidden_dim: int = 128
    latent_dim: int = 128
    dropout: float = 0.1
    tau: float = 0.07
    proj_bias: bool = False


class CLIPModel(nn.Module):
    def __init__(
        self,
        rna_dim: int = 64,
        prot_dim: int = 64,
        dproj: int = 256,
        depth: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        dropout: float = 0.1,
        tau: float = 0.07,
        proj_bias: bool = False,
    ):
        super().__init__()
        self.rna_head = RNAProjectionHead(rna_dim, dproj, bias=proj_bias)
        self.protein_tower = ProteinTower(
            in_dim=prot_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
            dproj=dproj, depth=depth, dropout=dropout, proj_bias=proj_bias,
        )
        self.tau = float(tau)   # température FIXE (cf. §5.2)
        self.dproj = dproj

    @classmethod
    def from_config(cls, cfg: CLIPConfig) -> "CLIPModel":
        return cls(**asdict(cfg))

    def forward(self, rna: torch.Tensor, protein: torch.Tensor):
        """rna (N, rna_dim), protein (N, prot_dim) → z_r, z_p (N, dproj), normalisés L2."""
        return self.rna_head(rna), self.protein_tower(protein)


# --------------------------------------------------------------------------- #
# Loss InfoNCE symétrique (CLIP)
# --------------------------------------------------------------------------- #
def info_nce_symmetric(z_r: torch.Tensor, z_p: torch.Tensor, tau: float):
    """z_r, z_p : (N, d) normalisés L2. La paire positive de la ligne i est la colonne i.

    Renvoie (loss, loss_ARN→prot, loss_prot→ARN) — les deux directions pour le monitoring.
    """
    logits = (z_r @ z_p.t()) / tau          # (N, N) = similarités cosinus / τ
    labels = torch.arange(z_r.size(0), device=z_r.device)
    loss_r2p = F.cross_entropy(logits, labels)       # chaque ARN retrouve sa protéine
    loss_p2r = F.cross_entropy(logits.t(), labels)   # chaque protéine retrouve son ARN
    return 0.5 * (loss_r2p + loss_p2r), loss_r2p, loss_p2r


# --------------------------------------------------------------------------- #
# Smoke test : python model.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    N, rna_dim, prot_dim = 32, 64, 64
    model = CLIPModel(rna_dim=rna_dim, prot_dim=prot_dim, dproj=256, depth=2)
    rna = torch.randn(N, rna_dim)
    prot = torch.randn(N, prot_dim)
    z_r, z_p = model(rna, prot)
    loss, l_r2p, l_p2r = info_nce_symmetric(z_r, z_p, model.tau)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"z_r {tuple(z_r.shape)}  z_p {tuple(z_p.shape)}  ||z_r||={z_r.norm(dim=-1).mean():.3f}")
    print(f"loss={loss.item():.4f}  (ARN→prot={l_r2p.item():.4f}, prot→ARN={l_p2r.item():.4f})")
    print(f"params entraînables : {n_params:,}")
    # une étape d'optim pour vérifier que ça rétropropage
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss.backward(); opt.step()
    print("backward + step OK ✓")
