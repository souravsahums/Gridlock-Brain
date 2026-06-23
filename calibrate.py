"""
GRIDLOCK BRAIN — External-Authority Face Validity (rules-compliant)
==================================================================
The submission rules forbid external datasets, so we CANNOT calibrate PCII
against third-party traffic speeds. Instead we calibrate against an authoritative
congestion signal that is ALREADY INSIDE the provided dataset: the Bengaluru
Traffic Police junction registry (`junction_name` -> official "BTP###" junctions,
i.e. the city's own designated, signal-controlled choke points).

Hypothesis: if PCII captures real congestion impact, its hotspots should land
disproportionately on these officially-recognised junctions.

Anti-circularity: PCII already contains a junction amplifier J. We therefore run
the test on **PCII_noJ** (PCII with J removed). If high-impact cells STILL coincide
with official junctions without the junction factor, the agreement is real, not
built in.
"""
from __future__ import annotations
import json
import os
import re

import numpy as np
import pandas as pd

import pipeline as P


def spearman(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    df = P.load()
    df = P.add_chronicity(df)
    df = P.compute_pcii(df)
    df = P.gridify(df)

    # PCII with the junction amplifier removed (J divides out cleanly)
    df["PCII_noJ"] = df["PCII"] / df["J"]

    g = df.groupby("cell")
    cells = pd.DataFrame({
        "pcii": g["PCII"].sum(),
        "pcii_noj": g["PCII_noJ"].sum(),
        "junction_share": g["at_junction"].mean(),
        "count": g.size(),
    })
    # a cell "is an official BTP junction" if most of its violations were tagged
    # to a named BTP junction by the enforcement system
    cells["is_official"] = cells["junction_share"] >= 0.5

    n = len(cells)
    base_rate = float(cells["is_official"].mean())

    # ---- Metric A: lift in the top-impact decile (NON-circular, uses PCII_noJ) ----
    k = max(1, n // 10)
    top_decile = cells.sort_values("pcii_noj", ascending=False).head(k)
    top_rate = float(top_decile["is_official"].mean())
    lift = top_rate / base_rate if base_rate else float("nan")

    # ---- Metric B: monotonic association (PCII_noJ vs official-junction share) ----
    rho_noj = spearman(cells["pcii_noj"], cells["junction_share"])
    rho_full = spearman(cells["pcii"], cells["junction_share"])

    # ---- Metric C: official-junction coincidence of the published top-60 hotspots
    top60 = cells.sort_values("pcii", ascending=False).head(60)
    top60_official = float(top60["is_official"].mean())
    top60_noj = cells.sort_values("pcii_noj", ascending=False).head(60)
    top60_noj_official = float(top60_noj["is_official"].mean())

    # ---- Metric D: recall of the city's busiest official junctions ----
    # distinct official BTP junctions ranked by violation volume
    jdf = df[df["at_junction"]].copy()
    jdf["bcode"] = jdf["junction_name"].str.extract(r"^(BTP\d+)")
    busiest = (jdf.groupby("bcode")["PCII_noJ"].sum()
               .sort_values(ascending=False).head(20).index)
    # which official junction does each top-60 hotspot cell sit on?
    top60_cells = set(top60.index)
    cell_junction = {}
    for cell, sub in df[df["at_junction"]].groupby("cell"):
        code = sub["junction_name"].str.extract(r"^(BTP\d+)")[0].mode()
        if len(code):
            cell_junction[cell] = code.iloc[0]
    covered = {cell_junction.get(c) for c in top60_cells} & set(busiest)
    recall_busy20 = len(covered) / len(busiest)

    out = {
        "method": "BTP official-junction registry (in-dataset authority); "
                  "junction factor removed from PCII to avoid circularity",
        "base_rate_official_pct": round(base_rate * 100, 1),
        "top_decile_official_pct": round(top_rate * 100, 1),
        "lift_top_decile": round(lift, 2),
        "spearman_pciiNoJ_vs_officialshare": round(rho_noj, 3),
        "spearman_pciiFull_vs_officialshare": round(rho_full, 3),
        "top60_official_pct": round(top60_official * 100, 1),
        "top60_noJ_official_pct": round(top60_noj_official * 100, 1),
        "recall_busiest20_official_junctions_pct": round(recall_busy20 * 100, 1),
    }
    os.makedirs(P.OUT, exist_ok=True)
    json.dump(out, open(os.path.join(P.OUT, "calibration.json"), "w"), indent=2)

    # inject into the dashboard bundle so the Model Trust card can show it
    dpath = os.path.join(P.HERE, "dashboard", "data.js")
    if os.path.exists(dpath):
        raw = open(dpath, encoding="utf-8").read()
        obj = json.loads(raw[len("window.GRID = "):-1])
        obj["calibration"] = out
        with open(dpath, "w", encoding="utf-8") as fh:
            fh.write("window.GRID = ")
            json.dump(obj, fh)
            fh.write(";")
        print("  -> injected calibration into dashboard/data.js")

    print("\n=== EXTERNAL-AUTHORITY FACE VALIDITY (rules-compliant) ===")
    print(f"  Base rate of official BTP junctions across all cells : {out['base_rate_official_pct']}%")
    print(f"  Among top-decile impact cells (PCII without J)       : {out['top_decile_official_pct']}%")
    print(f"  => Lift                                              : {out['lift_top_decile']}x")
    print(f"  Spearman(PCII without J, official-junction share)    : {out['spearman_pciiNoJ_vs_officialshare']}")
    print(f"  Top-60 hotspots sitting on an official BTP junction  : {out['top60_official_pct']}%")
    print(f"     (even with junction factor removed)               : {out['top60_noJ_official_pct']}%")
    print(f"  Recall of the 20 busiest official junctions          : {out['recall_busiest20_official_junctions_pct']}%")


if __name__ == "__main__":
    main()
