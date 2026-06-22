"""
GRIDLOCK BRAIN — PCII Validation & Robustness Suite
===================================================
The toughest judge question is: "Your impact weights are made up — why should we
trust PCII?"  This module answers that WITHOUT any external dataset, using five
defensible tests computed purely on the provided enforcement data:

  1. PREDICTIVE VALIDITY (temporal hold-out)
     Train PCII on the first 70% of the timeline, then check whether the hotspots
     it flags actually keep generating violations in the held-out last 30%.
     -> proves hotspots are STRUCTURAL & ACTIONABLE, not random noise.

  2. CONVERGENT VALIDITY (non-circular)
     PCII vs. each cell's number of DISTINCT ACTIVE DAYS (a persistence signal the
     PCII formula never sees). High correlation => high-PCII places are chronically
     obstructed, i.e. real congestion sources.

  3. WEIGHT ROBUSTNESS (Monte-Carlo sensitivity)
     Randomly perturb every severity & footprint weight by +/-25%, 100 times, and
     measure how much the hotspot RANKING moves (Spearman). Stable ranking =>
     conclusions are not an artefact of our exact weight choices.

  4. SPATIAL NON-RANDOMNESS (Moran's I + permutation p)
     Shows impact is significantly spatially clustered => justifies hotspot logic.

  5. CONCENTRATION (Gini + Pareto)
     Quantifies how targetable the problem is.

Outputs outputs/validation.json + a console report, and injects headline trust
metrics into the dashboard bundle.
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

import pipeline as P

OUT = P.OUT
RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# small stats helpers (no scipy dependency)
# --------------------------------------------------------------------------- #
def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def morans_i(cell_value: dict, radius: int = 1, perms: int = 199):
    """Global Moran's I with a binary distance-band weight, plus permutation p."""
    cells = list(cell_value.keys())
    x = np.array([cell_value[c] for c in cells], dtype=float)
    n = len(cells)
    z = x - x.mean()
    idx = {c: i for i, c in enumerate(cells)}
    # neighbour pairs (exclude self)
    rows, cols = [], []
    for c in cells:
        gx, gy = c
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (gx + dx, gy + dy)
                if nb in idx:
                    rows.append(idx[c]); cols.append(idx[nb])
    rows = np.array(rows); cols = np.array(cols)
    W = len(rows)
    denom = (z * z).sum()

    def I_of(zz):
        num = np.sum(zz[rows] * zz[cols])
        return (n / W) * (num / denom)

    I = I_of(z)
    # permutation null
    ge = 1
    for _ in range(perms):
        zp = RNG.permutation(z)
        if I_of(zp) >= I:
            ge += 1
    p = ge / (perms + 1)
    return float(I), float(p)


# --------------------------------------------------------------------------- #
def main():
    print("Loading + scoring (reuse pipeline) ...")
    df = P.load()
    df = P.add_chronicity(df)
    df = P.compute_pcii(df)
    df = P.gridify(df)

    # ---------- 1. PREDICTIVE VALIDITY (temporal hold-out) ----------
    print("\n[1] Predictive validity (70/30 temporal hold-out) ...")
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.70)]
    train = df[df["date"] <= cut]
    test = df[df["date"] > cut]
    tr_pcii = train.groupby("cell")["PCII"].sum()
    te_cnt = test.groupby("cell").size()
    common = tr_pcii.index.intersection(te_cnt.index)
    a = tr_pcii.loc[common].to_numpy()
    b = te_cnt.loc[common].to_numpy()
    sp_pred = spearman(a, b)

    # precision@K and impact capture: do train-top-K cells dominate test volume?
    K = 50
    top_train = set(tr_pcii.sort_values(ascending=False).head(K).index)
    top_test = set(te_cnt.sort_values(ascending=False).head(K).index)
    precision_at_k = len(top_train & top_test) / K
    test_total = te_cnt.sum()
    capture = te_cnt.reindex(list(top_train)).fillna(0).sum() / test_total

    print(f"    Spearman(train PCII, test violations) = {sp_pred:.3f}")
    print(f"    Precision@{K} (hotspots persist)       = {precision_at_k:.0%}")
    print(f"    Top-{K} train hotspots capture {capture:.0%} of all FUTURE violations")

    # ---------- 2. CONVERGENT VALIDITY (non-circular) ----------
    print("\n[2] Convergent validity vs. distinct active days ...")
    active_days = df.groupby("cell")["date"].nunique()
    cell_pcii = df.groupby("cell")["PCII"].sum()
    c = cell_pcii.index
    sp_conv = spearman(cell_pcii.loc[c].to_numpy(), active_days.loc[c].to_numpy())
    print(f"    Spearman(PCII, distinct active days)   = {sp_conv:.3f}")

    # ---------- 3. WEIGHT ROBUSTNESS (Monte-Carlo) ----------
    print("\n[3] Weight robustness (100x +/-25% perturbation) ...")

    def sev_label(vlist):
        if not vlist:
            return "__default"
        return max(vlist, key=lambda v: P.SEVERITY.get(v, P.DEFAULT_SEVERITY))

    df["_sevlab"] = df["vlist"].map(sev_label)
    sev_labels = sorted(df["_sevlab"].unique())
    veh_labels = sorted(df["vehicle_type"].astype(str).unique())
    sl_code = {l: i for i, l in enumerate(sev_labels)}
    vl_code = {l: i for i, l in enumerate(veh_labels)}
    sev_code_arr = df["_sevlab"].map(sl_code).to_numpy()
    veh_code_arr = df["vehicle_type"].astype(str).map(vl_code).to_numpy()
    rest = (P.BASE * df["J"] * df["T"] * (1 + df["C"])).to_numpy()
    cell_codes, cell_uniques = pd.factorize(df["cell"])
    base_sev = np.array([P.SEVERITY.get(l, P.DEFAULT_SEVERITY) if l != "__default"
                         else P.DEFAULT_SEVERITY for l in sev_labels])
    base_foot = np.array([P.FOOTPRINT.get(l, P.DEFAULT_FOOTPRINT) for l in veh_labels])

    base_cell = np.bincount(cell_codes,
                            weights=base_sev[sev_code_arr] * base_foot[veh_code_arr] * rest)
    sp_runs = []
    for _ in range(100):
        ps = base_sev * (1 + RNG.uniform(-0.25, 0.25, size=base_sev.shape))
        pf = base_foot * (1 + RNG.uniform(-0.25, 0.25, size=base_foot.shape))
        pcii = ps[sev_code_arr] * pf[veh_code_arr] * rest
        cell_tot = np.bincount(cell_codes, weights=pcii)
        sp_runs.append(spearman(base_cell, cell_tot))
    sp_runs = np.array(sp_runs)
    print(f"    Spearman(baseline, perturbed) mean={sp_runs.mean():.3f} "
          f"min={sp_runs.min():.3f}")

    # ranking stability of the TOP-50 specifically
    base_top = set(np.argsort(base_cell)[::-1][:50])
    overlaps = []
    for _ in range(100):
        ps = base_sev * (1 + RNG.uniform(-0.25, 0.25, size=base_sev.shape))
        pf = base_foot * (1 + RNG.uniform(-0.25, 0.25, size=base_foot.shape))
        pcii = ps[sev_code_arr] * pf[veh_code_arr] * rest
        cell_tot = np.bincount(cell_codes, weights=pcii)
        pt = set(np.argsort(cell_tot)[::-1][:50])
        overlaps.append(len(base_top & pt) / 50)
    top_stability = float(np.mean(overlaps))
    print(f"    Top-50 hotspot stability under perturbation = {top_stability:.0%}")

    # ---------- 4. SPATIAL NON-RANDOMNESS (Moran's I) ----------
    print("\n[4] Spatial autocorrelation (Moran's I) ...")
    I, p = morans_i(cell_pcii.to_dict(), radius=1, perms=199)
    print(f"    Moran's I = {I:.3f}  (permutation p = {p:.3f})")

    # ---------- 5. CONCENTRATION ----------
    print("\n[5] Concentration ...")
    vals = cell_pcii.to_numpy()
    g = gini(vals)
    s = np.sort(vals)[::-1]
    tot = s.sum()
    p1 = s[:max(1, len(s)//100)].sum() / tot
    p5 = s[:max(1, len(s)//20)].sum() / tot
    p10 = s[:max(1, len(s)//10)].sum() / tot
    print(f"    Gini = {g:.3f} | top1%={p1:.0%} top5%={p5:.0%} top10%={p10:.0%}")

    # ---------- write ----------
    val = {
        "predictive": {
            "spearman_train_pcii_vs_test_violations": round(sp_pred, 3),
            "precision_at_50": round(precision_at_k, 3),
            "top50_future_violation_capture": round(float(capture), 3),
            "train_period_end": str(cut),
        },
        "convergent": {"spearman_pcii_vs_active_days": round(sp_conv, 3)},
        "robustness": {
            "spearman_mean": round(float(sp_runs.mean()), 3),
            "spearman_min": round(float(sp_runs.min()), 3),
            "top50_stability": round(top_stability, 3),
            "perturbation": "+/-25% on all severity & footprint weights, n=100",
        },
        "spatial": {"morans_I": round(I, 3), "permutation_p": round(p, 3)},
        "concentration": {
            "gini": round(g, 3),
            "top1pct": round(float(p1), 3),
            "top5pct": round(float(p5), 3),
            "top10pct": round(float(p10), 3),
        },
    }
    json.dump(val, open(os.path.join(OUT, "validation.json"), "w"), indent=2)

    # inject into dashboard bundle (data.js) so the demo can show a Trust panel
    dpath = os.path.join(P.HERE, "dashboard", "data.js")
    if os.path.exists(dpath):
        raw = open(dpath, encoding="utf-8").read()
        obj = json.loads(raw[len("window.GRID = "):-1])
        obj["validation"] = val
        with open(dpath, "w", encoding="utf-8") as fh:
            fh.write("window.GRID = ")
            json.dump(obj, fh)
            fh.write(";")
        print("\n  -> injected validation into dashboard/data.js")

    print("\n=== VALIDATION SUMMARY (cite these in the deck) ===")
    print(f"  Predictive: train hotspots predict future violations  Spearman={sp_pred:.2f}, "
          f"precision@50={precision_at_k:.0%}, capture {capture:.0%} of future tickets")
    print(f"  Convergent: PCII vs chronic persistence (active days) Spearman={sp_conv:.2f}")
    print(f"  Robust:     top-50 hotspots {top_stability:.0%} stable under +/-25% weight noise")
    print(f"  Spatial:    Moran's I={I:.2f} (p={p:.3f}) -> non-random clustering")
    print(f"  Targetable: Gini={g:.2f}, top1% of cells = {p1:.0%} of impact")


if __name__ == "__main__":
    main()
