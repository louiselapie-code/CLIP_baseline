"""
Runner CPU-friendly pour le CLIP ARN-protéine de Louise.

IMPORTANT : ce script n'invente AUCUNE science. Il importe directement le code du
repo (src/) :
    - CLIPModel, CLIPConfig, info_nce_symmetric        (model.py)
    - set_seed                                         (protein_encoder.py)
    - cca_fit, cca_transform, cross_modal_retrieval, random_floor (sanity_check.py)
La boucle d'entraînement reproduit fidèlement train.py (AdamW + cosine decay + warmup,
grad clip, dropout, early stopping sur R@5 val). Seules DEUX adaptations, pour tenir
dans un sandbox CPU avec commandes plafonnées à 45 s :
    1. la R@5 de validation EN BOUCLE est calculée sur un sous-échantillon (--val-sample)
       — uniquement pour le monitoring / early stopping ;
    2. l'évaluation finale se fait sur une galerie de taille fixe (--eval-gallery) tirée
       du TEST, identique pour CLIP / CCA / floor (comparaison équitable) et IDENTIQUE
       entre CosMx et Xenium (donc directement comparables).
Les checkpoints sont compatibles avec evaluate.py de Louise.

Modes : --mode train | eval | both
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def import_src(src_dir):
    sys.path.insert(0, str(src_dir))
    import model as M
    import protein_encoder as PE
    import sanity_check as SC
    return M, PE, SC


def load_views(paired_dir):
    d = Path(paired_dir)
    rna = np.load(d / "paired_rna.npy").astype(np.float32)
    prot = np.load(d / "paired_protein.npy").astype(np.float32)
    cells = pd.read_csv(d / "paired_cells.csv")
    return rna, prot, cells["cell_id"].astype(str).to_numpy(), cells["split"].astype(str).to_numpy()


def make_scheduler(opt, total_steps, warmup):
    def fn(step):
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def embed_all(model, rna, prot, device, bs=16384):
    model.eval(); zr, zp = [], []
    for s in range(0, len(rna), bs):
        a, b = model(torch.from_numpy(rna[s:s+bs]).to(device),
                     torch.from_numpy(prot[s:s+bs]).to(device))
        zr.append(a.float().cpu().numpy()); zp.append(b.float().cpu().numpy())
    return np.concatenate(zr), np.concatenate(zp)


def r5_mean(SC, Zr, Zp):
    r_ab, _, _ = SC.cross_modal_retrieval(Zr, Zp, ks=(1, 5))
    r_ba, _, _ = SC.cross_modal_retrieval(Zp, Zr, ks=(1, 5))
    return 0.5 * (r_ab[5] + r_ba[5])


def train(args, M, PE, SC):
    PE.set_seed(args.seed)
    device = torch.device(args.device)
    rna, prot, cid, split = load_views(args.paired_dir)
    tr, va = split == "train", split == "val"
    rna_tr, prot_tr = rna[tr], prot[tr]
    rna_va, prot_va = rna[va], prot[va]
    # sous-échantillon de val pour le monitoring rapide
    rng = np.random.default_rng(args.seed)
    if args.val_sample and va.sum() > args.val_sample:
        idx = rng.choice(va.sum(), size=args.val_sample, replace=False)
        rna_vm, prot_vm = rna_va[idx], prot_va[idx]
    else:
        rna_vm, prot_vm = rna_va, prot_va
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device} train={tr.sum()} val={va.sum()} "
          f"(monitor sur {len(rna_vm)}) rna_dim={rna.shape[1]} prot_dim={prot.shape[1]}")

    cfg = M.CLIPConfig(rna_dim=rna.shape[1], prot_dim=prot.shape[1], dproj=args.dproj,
                       depth=args.depth, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                       dropout=args.dropout, tau=args.tau)
    model = M.CLIPModel.from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(torch.from_numpy(rna_tr), torch.from_numpy(prot_tr))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=len(ds) > args.batch_size)
    total_steps = max(1, len(dl)) * args.epochs
    sched = make_scheduler(opt, total_steps, args.warmup)
    json.dump({**cfg.__dict__, "lr": args.lr, "weight_decay": args.weight_decay,
               "batch_size": args.batch_size, "epochs": args.epochs, "seed": args.seed,
               "val_sample": args.val_sample}, open(out / "config.json", "w"), indent=2)

    # --- reprise éventuelle (entraînement par tranches pour tenir dans 45 s/commande) ---
    state_path = out / "trainer_state.pt"
    done_path = out / "TRAIN_DONE"
    best, wait, rows, start_epoch = -1.0, 0, [], 1
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, wait, rows, start_epoch = st["best"], st["wait"], st["rows"], st["epoch"] + 1
        try:
            torch.set_rng_state(st["rng_torch"]); np.random.set_state(st["rng_np"])
        except Exception:
            pass
        print(f"[train] reprise à l'époque {start_epoch} (best R@5 monitor={100*best:.2f}%)")
    if done_path.exists() and args.resume:
        print("[train] TRAIN_DONE présent -> entraînement déjà terminé, rien à faire."); return

    end_epoch = args.epochs if not args.chunk_epochs else min(args.epochs, start_epoch + args.chunk_epochs - 1)
    t0 = time.time(); finished = False; ep = start_epoch - 1
    for ep in range(start_epoch, end_epoch + 1):
        model.train(); tl = nb = 0
        for r, p in dl:
            r, p = r.to(device), p.to(device)
            z_r, z_p = model(r, p)
            loss, lr2p, lp2r = M.info_nce_symmetric(z_r, z_p, model.tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            tl += loss.item(); nb += 1
        Zr, Zp = embed_all(model, rna_vm, prot_vm, device)
        r5 = r5_mean(SC, Zr, Zp)
        pos = float(np.mean(np.sum(Zr * Zp, axis=1)))
        std = float(Zr.std(0).mean())
        rows.append({"epoch": ep, "train_loss": tl / nb, "val_R5_mean": r5,
                     "pos_cos": pos, "std_rna": std, "lr": opt.param_groups[0]["lr"]})
        print(f"[{ep:3d}/{args.epochs}] loss={tl/nb:.4f} valR@5={100*r5:.2f}% posCos={pos:.3f} std={std:.3f}")
        if r5 > best:
            best = r5; wait = 0
            torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                        "epoch": ep, "val_R5_mean": r5}, out / "best.pt")
        else:
            wait += 1
        # sauvegarde de l'état (reprise) + logs à CHAQUE époque
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "wait": wait, "rows": rows,
                    "rng_torch": torch.get_rng_state(), "rng_np": np.random.get_state()}, state_path)
        with open(out / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        _plot(out / "training_curves.png", rows)
        if wait >= args.patience:
            print(f"[train] early stop ep {ep} (pas de gain R@5 depuis {args.patience})"); finished = True; break
    if ep >= args.epochs:
        finished = True
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "epoch": ep}, out / "last.pt")
    if finished:
        done_path.write_text(f"done at epoch {ep}, best R@5 monitor={best:.4f}\n")
        print(f"[train] TERMINÉ à l'époque {ep} en {time.time()-t0:.0f}s (cette tranche), best R@5(monitor)={100*best:.2f}%")
    else:
        print(f"[train] tranche faite jusqu'à l'époque {ep}/{args.epochs} en {time.time()-t0:.0f}s "
              f"(relancer avec --resume pour continuer)")


def _plot(path, rows):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [r["train_loss"] for r in rows]); ax[0].set_title("Loss train"); ax[0].set_xlabel("époque")
    ax[1].plot(ep, [100*r["val_R5_mean"] for r in rows]); ax[1].set_title("R@5 val (monitor, %)"); ax[1].set_xlabel("époque")
    ax[2].plot(ep, [r["pos_cos"] for r in rows], label="cos paires +")
    ax[2].plot(ep, [r["std_rna"] for r in rows], label="std/dim ARN"); ax[2].legend()
    ax[2].set_title("Alignement & collapse"); ax[2].set_xlabel("époque")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def fmt_ret(SC, A, B, ks=(1, 5, 10, 50)):
    r_ab, m_ab, N = SC.cross_modal_retrieval(A, B, ks=ks)
    r_ba, m_ba, _ = SC.cross_modal_retrieval(B, A, ks=ks)
    return {"ARN→prot": r_ab, "MedR_ab": m_ab, "prot→ARN": r_ba, "MedR_ba": m_ba, "N": N}


def evaluate(args, M, PE, SC):
    device = torch.device(args.device)
    rna, prot, cid, split = load_views(args.paired_dir)
    tr, te = split == "train", split == "test"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(Path(args.outdir) / "best.pt", map_location="cpu", weights_only=False)
    model = M.CLIPModel.from_config(M.CLIPConfig(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"])

    # galerie test de taille fixe (équitable + comparable entre tissus)
    rng = np.random.default_rng(0)
    te_idx = np.where(te)[0]
    if args.eval_gallery and len(te_idx) > args.eval_gallery:
        te_idx = np.sort(rng.choice(te_idx, size=args.eval_gallery, replace=False))
    rna_te, prot_te = rna[te_idx], prot[te_idx]
    N = len(te_idx)
    print(f"[eval] test gallery N={N} (sur {te.sum()} test) | train={tr.sum()}")

    Zr, Zp = embed_all(model, rna_te, prot_te, device)
    rep = {"N_gallery": N, "n_test_total": int(te.sum()), "n_train": int(tr.sum())}

    rec_f, medr_f = SC.random_floor(N)
    rep["floor"] = {"recall": rec_f, "MedR": medr_f}
    # CCA fit sur train complet
    cca = SC.cca_fit(rna[tr], prot[tr], args.cca_components)
    A, B = SC.cca_transform(cca, rna_te, prot_te)
    rep["CCA"] = fmt_ret(SC, A, B)
    rep["CLIP"] = fmt_ret(SC, Zr, Zp)
    perm = rng.permutation(N)
    rep["CLIP_permute"] = fmt_ret(SC, Zr, Zp[perm])
    rep["cca_corr_top5"] = [float(x) for x in cca["corr"][:5]]
    # diagnostics
    s = min(4000, N); ii = rng.choice(N, size=s, replace=False)
    def offdiag(z):
        S = z @ z.T; n = len(z); return float((S.sum() - np.trace(S)) / (n*(n-1)))
    rep["diagnostics"] = {"pos_cos": float(np.mean(np.sum(Zr*Zp, axis=1))),
                          "intra_cos_rna": offdiag(Zr[ii]), "intra_cos_prot": offdiag(Zp[ii]),
                          "std_rna": float(Zr.std(0).mean()), "std_prot": float(Zp.std(0).mean())}

    print(f"  floor   R@1/5/10/50 = {100*rec_f[1]:.3f}/{100*rec_f[5]:.3f}/{100*rec_f[10]:.3f}/{100*rec_f[50]:.3f}%  MedR={medr_f:.0f}")
    for nm in ["CCA", "CLIP", "CLIP_permute"]:
        d = rep[nm]
        print(f"  {nm:13s} ARN→prot R@1/5/10/50 = "
              f"{100*d['ARN→prot'][1]:.2f}/{100*d['ARN→prot'][5]:.2f}/{100*d['ARN→prot'][10]:.2f}/{100*d['ARN→prot'][50]:.2f}%  MedR={d['MedR_ab']:.0f}")
        print(f"  {'':13s} prot→ARN R@1/5/10/50 = "
              f"{100*d['prot→ARN'][1]:.2f}/{100*d['prot→ARN'][5]:.2f}/{100*d['prot→ARN'][10]:.2f}/{100*d['prot→ARN'][50]:.2f}%  MedR={d['MedR_ba']:.0f}")

    clip_r5 = 0.5*(rep["CLIP"]["ARN→prot"][5] + rep["CLIP"]["prot→ARN"][5])
    cca_r5 = 0.5*(rep["CCA"]["ARN→prot"][5] + rep["CCA"]["prot→ARN"][5])
    rep["verdict_clip_gt_cca"] = bool(clip_r5 > cca_r5)
    json.dump(rep, open(out / "eval_report.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[eval] R@5 test (moy.) CLIP={100*clip_r5:.2f}% vs CCA={100*cca_r5:.2f}%  -> "
          f"{'CLIP BAT CCA ✓' if clip_r5>cca_r5 else 'CLIP ne bat pas CCA ✗'}")
    print(f"[eval] rapport (retrieval) -> {out}/eval_report.json")


def eval_labels(args, M, PE, SC):
    """Passe séparée : clustering (ARI/NMI) + sonde linéaire, fusionnée dans eval_report.json."""
    device = torch.device(args.device)
    rna, prot, cid, split = load_views(args.paired_dir)
    tr, te = split == "train", split == "test"
    out = Path(args.outdir)
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model = M.CLIPModel.from_config(M.CLIPConfig(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    rep = json.load(open(out / "eval_report.json")) if (out / "eval_report.json").exists() else {}
    rng = np.random.default_rng(0)

    df = pd.read_csv(args.labels)
    idc = next((c for c in df.columns if c.lower() in ("cell_id", "cellid")), df.columns[0])
    lc = args.label_col if args.label_col in df.columns else df.columns[1]
    mp = dict(zip(df[idc].astype(str), df[lc].astype(str)))
    lab = np.array([mp.get(c, "NA") for c in cid], dtype=object)
    has = lab != "NA"
    rep["label_coverage"] = float(has.mean())
    rep["label_types"] = sorted(set(lab[has].tolist()))
    k = args.n_clusters or len(rep["label_types"])
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    # test sous-échantillonné (galerie) + train sous-échantillonné (sonde)
    te_idx = np.where(te)[0]
    if args.eval_gallery and len(te_idx) > args.eval_gallery:
        te_idx = np.sort(rng.choice(te_idx, size=args.eval_gallery, replace=False))
    lab_te = lab[te_idx]; te_has = lab_te != "NA"
    tr_idx = np.where(tr)[0]; lab_tr = lab[tr_idx]; tr_has = lab_tr != "NA"
    tr_keep = np.where(tr_has)[0]
    if len(tr_keep) > args.probe_train:
        tr_keep = rng.choice(tr_keep, size=args.probe_train, replace=False)
    Zr_te, Zp_te = embed_all(model, rna[te_idx], prot[te_idx], device)
    Zr_tr, Zp_tr = embed_all(model, rna[tr_idx][tr_keep], prot[tr_idx][tr_keep], device)
    y_tr = lab_tr[tr_keep]; y_te = lab_te[te_has]
    spaces = {"CLIP joint": (l2(Zr_tr+Zp_tr), l2(Zr_te[te_has]+Zp_te[te_has])),
              "CLIP ARN": (Zr_tr, Zr_te[te_has]),
              "CLIP protéine": (Zp_tr, Zp_te[te_has]),
              "NOVAE brut": (rna[tr_idx][tr_keep], rna[te_idx][te_has]),
              "protéine brute": (prot[tr_idx][tr_keep], prot[te_idx][te_has])}
    clu, prb = {}, {}
    print(f"[labels] {k} types, couverture {100*has.mean():.0f}%, test={te_has.sum()}, probe-train={len(tr_keep)}")
    print(f"  {'espace':16s}{'ARI':>7}{'NMI':>7}{'acc':>8}{'F1':>7}")
    for nm, (Ztr, Zte) in spaces.items():
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(Zte)
        ari = float(adjusted_rand_score(y_te, km)); nmi = float(normalized_mutual_info_score(y_te, km))
        sc = StandardScaler().fit(Ztr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Ztr), y_tr)
        pred = clf.predict(sc.transform(Zte))
        acc = float(accuracy_score(y_te, pred)); f1 = float(f1_score(y_te, pred, average="macro"))
        clu[nm] = {"ARI": ari, "NMI": nmi}; prb[nm] = {"acc": acc, "f1": f1}
        print(f"  {nm:16s}{ari:7.3f}{nmi:7.3f}{acc:8.3f}{f1:7.3f}")
    rep["clustering"] = clu; rep["linear_probe"] = prb
    json.dump(rep, open(out / "eval_report.json", "w"), indent=2, ensure_ascii=False)
    print(f"[labels] fusionné -> {out}/eval_report.json")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--paired-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--mode", choices=["train", "results/eval", "labels", "both"], default="both")
    p.add_argument("--labels", default=None)
    p.add_argument("--label-col", default="qc_celltype_cpu")
    p.add_argument("--n-clusters", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--dproj", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true", help="reprend depuis trainer_state.pt si présent")
    p.add_argument("--chunk-epochs", type=int, default=0, help="nb max d'époques par invocation (0 = tout)")
    p.add_argument("--val-sample", type=int, default=4000)
    p.add_argument("--eval-gallery", type=int, default=18944)
    p.add_argument("--cca-components", type=int, default=32)
    p.add_argument("--probe-train", type=int, default=15000)
    return p.parse_args()


if __name__ == "__main__":
    a = parse()
    M, PE, SC = import_src(a.src)
    torch.set_num_threads(4)
    if a.mode in ("train", "both"):
        train(a, M, PE, SC)
    if a.mode in ("results/eval", "both"):
        evaluate(a, M, PE, SC)
    if a.mode in ("labels", "both") and a.labels:
        eval_labels(a, M, PE, SC)
