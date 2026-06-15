"""Figures pour le rapport/slides à partir des eval_report.json + checkpoints réels."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("/sessions/keen-adoring-galileo/mnt/CLIP_baseline_v0")
OUT = Path("/sessions/keen-adoring-galileo/mnt/outputs/figs"); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(P / "src"))
import torch, model as M

TISSUES = [("CosMx breast", "clip_cosmx_seed42", "data/processed/cosmx_breast",
            "data/raw/cosmx_breast/tables/cosmx_breast_celltype_annotations_seed42.csv"),
           ("Xenium renal", "clip_xenium_seed42", "data/processed/xenium_renal",
            "data/raw/xenium_renal/tables/xenium_renal_celltype_annotations_seed42.csv")]
C_FLOOR, C_CCA, C_CLIP = "#9aa0a6", "#e8833a", "#2a7de1"
reports = {t: json.load(open(P / "runs" / run / "eval_report.json")) for t, run, _, _ in TISSUES}

# ---------- 1. Retrieval R@5 (moyenne 2 sens) + MedR ----------
def r5mean(d): return 50*(d["ARN→prot"]["5"] + d["prot→ARN"]["5"])
def medrmean(d): return 0.5*(d["MedR_ab"] + d["MedR_ba"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
labels = [t for t, *_ in TISSUES]; x = np.arange(len(labels)); w = 0.26
ax[0].bar(x-w, [100*reports[t]["floor"]["recall"]["5"] for t in labels], w, label="Hasard (floor)", color=C_FLOOR)
ax[0].bar(x,   [r5mean(reports[t]["CCA"]) for t in labels], w, label="CCA (linéaire)", color=C_CCA)
ax[0].bar(x+w, [r5mean(reports[t]["CLIP"]) for t in labels], w, label="CLIP (le nôtre)", color=C_CLIP)
for i, t in enumerate(labels):
    ax[0].text(i, r5mean(reports[t]["CLIP"])+0.02, f"{r5mean(reports[t]['CLIP']):.2f}%", ha="center", fontsize=9, color=C_CLIP, fontweight="bold")
    ax[0].text(i, r5mean(reports[t]["CCA"])+0.02, f"{r5mean(reports[t]['CCA']):.2f}%", ha="center", fontsize=8, color=C_CCA)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("Recall@5 (%)")
ax[0].set_title("Recall@5 cross-modal sur le TEST (galerie 15 000)\nplus haut = mieux"); ax[0].legend(frameon=False)
ax[1].bar(x-w, [reports[t]["floor"]["MedR"] for t in labels], w, label="Hasard", color=C_FLOOR)
ax[1].bar(x,   [medrmean(reports[t]["CCA"]) for t in labels], w, label="CCA", color=C_CCA)
ax[1].bar(x+w, [medrmean(reports[t]["CLIP"]) for t in labels], w, label="CLIP", color=C_CLIP)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylabel("Rang médian (MedR)")
ax[1].set_title("Rang médian du bon appariement\nplus bas = mieux"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT/"fig_retrieval.png", dpi=130); plt.close(fig)
print("fig_retrieval.png OK")

# ---------- 2. Sonde linéaire (F1 macro) par espace ----------
spaces = ["CLIP joint", "CLIP ARN", "CLIP protéine", "NOVAE brut", "protéine brute"]
cols = {"CLIP joint": C_CLIP, "CLIP ARN": "#7fb3f0", "CLIP protéine": "#5b9bd5",
        "NOVAE brut": "#bbbbbb", "protéine brute": "#d9a066"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for j, t in enumerate(labels):
    f1 = [reports[t]["linear_probe"][s]["f1"] for s in spaces]
    xx = np.arange(len(spaces))
    bars = ax[j].bar(xx, f1, color=[cols[s] for s in spaces])
    for b, v in zip(bars, f1):
        ax[j].text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=8)
    ax[j].set_xticks(xx); ax[j].set_xticklabels(spaces, rotation=25, ha="right", fontsize=8)
    ax[j].set_ylim(0, max(f1)*1.18); ax[j].set_ylabel("F1 macro (sonde linéaire)")
    ax[j].set_title(f"{t} — séparabilité des types cellulaires")
fig.suptitle("Sonde linéaire sur le TEST : l'espace CLIP joint est le plus informatif", y=1.02, fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"fig_probe.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("fig_probe.png OK")

# ---------- 3. PCA 2D de l'espace joint coloré par type + par modalité ----------
@torch.no_grad()
def embed(mdl, rna, prot):
    a, b = mdl(torch.from_numpy(rna), torch.from_numpy(prot)); return a.numpy(), b.numpy()
def l2(x): return x/np.clip(np.linalg.norm(x,axis=1,keepdims=True),1e-12,None)
def pca2(X):
    Xc=X-X.mean(0); U,S,Vt=np.linalg.svd(Xc,full_matrices=False); return Xc@Vt[:2].T
for t, run, pdir, lab in TISSUES:
    rna=np.load(P/pdir/"paired_rna.npy").astype(np.float32)
    prot=np.load(P/pdir/"paired_protein.npy").astype(np.float32)
    cells=pd.read_csv(P/pdir/"paired_cells.csv"); cid=cells["cell_id"].astype(str).to_numpy(); split=cells["split"].astype(str).to_numpy()
    df=pd.read_csv(P/lab); idc=next((c for c in df.columns if c.lower() in ("cell_id","cellid")), df.columns[0])
    mp=dict(zip(df[idc].astype(str), df["qc_celltype_cpu"].astype(str)))
    labels_all=np.array([mp.get(c,"NA") for c in cid],dtype=object)
    te=np.where(split=="test")[0]; rng=np.random.default_rng(0); sel=rng.choice(te,size=min(5000,len(te)),replace=False)
    ck=torch.load(P/"runs"/run/"best.pt",map_location="cpu",weights_only=False)
    mdl=M.CLIPModel.from_config(M.CLIPConfig(**ck["config"])).eval(); mdl.load_state_dict(ck["model"])
    zr,zp=embed(mdl,rna[sel],prot[sel]); y=labels_all[sel]
    joint=l2(zr+zp); XY=pca2(joint)
    stacked=np.vstack([zr,zp]); XY2=pca2(stacked); n=len(zr)
    fig, ax=plt.subplots(1,2,figsize=(12,5))
    cats=sorted(set(y.tolist())); cmap=plt.get_cmap("tab10")
    for i,c in enumerate(cats):
        m=y==c; ax[0].scatter(XY[m,0],XY[m,1],s=6,alpha=0.6,color=cmap(i%10),label=c)
    ax[0].set_title(f"{t} — espace CLIP joint (PCA), couleur = type cellulaire")
    ax[0].legend(markerscale=2,fontsize=7,loc="best",frameon=False); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].scatter(XY2[:n,0],XY2[:n,1],s=6,alpha=0.4,color=C_CLIP,label="ARN (z_r)")
    ax[1].scatter(XY2[n:,0],XY2[n:,1],s=6,alpha=0.4,color=C_CCA,label="protéine (z_p)")
    ax[1].set_title("Recouvrement ARN/protéine = alignement"); ax[1].legend(frameon=False); ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.tight_layout(); fn=OUT/f"fig_pca_{run.split('_')[1]}.png"; fig.savefig(fn,dpi=130); plt.close(fig)
    print(fn.name,"OK")
print("DONE figs ->", OUT)
