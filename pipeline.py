"""
GRIDLOCK BRAIN — Parking-Induced Congestion Intelligence Engine
================================================================
Pipeline: raw enforcement records  ->  decision-ready intelligence.

Novel core: PCII (Parking Congestion Impact Index)
--------------------------------------------------
A physics-informed score that converts a single parking ticket into an
ESTIMATE OF CARRIAGEWAY DISRUPTION *without any live traffic sensor*, by fusing:
  S_v : lane-blocking severity of the violation type (main road / double / crossing ...)
  F_v : physical footprint of the vehicle (a lorry blocks ~6x a scooter)
  J   : junction-proximity amplifier (blocking near an intersection cascades)
  T   : temporal coupling with baseline traffic demand (rush-hour overlap)
  C   : chronicity amplifier (repeat offenders / persistent black-spots)

    PCII = BASE * S_v * F_v * J * T * (1 + C)

We then:
  1. Aggregate PCII onto a ~220 m spatial grid.
  2. Run GETIS-ORD Gi* to find *statistically significant* hotspots (z-score),
     separating real choke-clusters from random noise.
  3. Build a 24x7 spatio-temporal demand profile per hotspot -> WHEN to deploy.
  4. Solve a greedy max-coverage PATROL OPTIMIZER -> WHERE to deploy K units
     for maximum congestion relief.

Outputs compact JSON consumed by the dashboard.
"""

from __future__ import annotations
import json
import math
import os
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# Dataset location is configurable so the repo runs anywhere:
#   1. env var  GRIDLOCK_CSV=/path/to/violations.csv
#   2. ./data/violations.csv          (recommended for a fresh clone)
#   3. ../<original dataset filename>  (original dev layout)
DATASET_NAME = "jan to may police violation_anonymized791b166.csv"


def find_csv():
    candidates = [
        os.environ.get("GRIDLOCK_CSV"),
        os.path.join(HERE, "data", "violations.csv"),
        os.path.join(HERE, "data", DATASET_NAME),
        os.path.join(HERE, DATASET_NAME),
        os.path.join(HERE, "..", DATASET_NAME),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Dataset CSV not found. Set GRIDLOCK_CSV, or place the file at "
        "solution/data/violations.csv. Looked in: "
        + "; ".join(str(c) for c in candidates if c))

CELL_DEG = 0.002          # ~220 m grid cell
GISTAR_RADIUS = 2         # neighbour band in cells for Gi*
N_PATROLS = 25            # patrol units to allocate in the optimizer
TOP_HOTSPOTS = 60

# --- Severity weights: how much each violation type blocks the carriageway ---
# Calibrated on first-principles lane-blocking impact (1.0 = baseline no-parking).
SEVERITY = {
    "DOUBLE PARKING": 2.6,                 # blocks a live lane outright
    "PARKING IN A MAIN ROAD": 2.3,         # arterial throughput hit
    "PARKING NEAR ROAD CROSSING": 2.2,     # sightline + turning-radius choke
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 2.0,
    "WRONG PARKING": 1.6,                  # partial obstruction
    "PARKING ON FOOTPATH": 1.3,            # pushes pedestrians into road
    "NO PARKING": 1.0,                     # baseline
    "DEFECTIVE NUMBER PLATE": 0.0,         # not a congestion contributor
    "REFUSE TO GO FOR HIRE": 0.0,
}
DEFAULT_SEVERITY = 1.2

# --- Vehicle footprint weights: ~proportional to occupied carriageway length ---
FOOTPRINT = {
    "HGV": 3.2, "LORRY/GOODS VEHICLE": 3.0, "BUS (BMTC/KSRTC)": 3.0,
    "PRIVATE BUS": 3.0, "TANKER": 3.0, "TEMPO": 2.0, "LGV": 1.9, "VAN": 1.8,
    "GOODS AUTO": 1.6, "MAXI-CAB": 1.6, "JEEP": 1.5, "CAR": 1.4,
    "PASSENGER AUTO": 1.2, "MOTOR CYCLE": 0.5, "SCOOTER": 0.5,
    "MOPED": 0.5, "BICYCLE": 0.3,
}
DEFAULT_FOOTPRINT = 1.3

BASE = 10.0  # scaling so scores read as friendly integers


# --------------------------------------------------------------------------- #
# 1. Load & feature-engineer
# --------------------------------------------------------------------------- #
def parse_violations(s: str) -> list[str]:
    """violation_type is a JSON-ish array string: ["WRONG PARKING","NO PARKING"]."""
    if not isinstance(s, str):
        return []
    return re.findall(r'"([^"]+)"', s)


def severity_of(vlist: list[str]) -> float:
    """Severity of a record = max single-violation severity (worst obstruction wins)."""
    if not vlist:
        return DEFAULT_SEVERITY
    return max(SEVERITY.get(v, DEFAULT_SEVERITY) for v in vlist)


def load() -> pd.DataFrame:
    print("Loading CSV ...")
    df = pd.read_csv(find_csv(), low_memory=False, usecols=[
        "id", "latitude", "longitude", "location", "vehicle_number",
        "vehicle_type", "violation_type", "created_datetime",
        "police_station", "junction_name",
    ])
    # clean geo
    df = df[(df.latitude.between(12.7, 13.4)) & (df.longitude.between(77.3, 77.9))].copy()

    # time -> IST
    dt = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df[dt.notna()].copy()
    ist = dt[dt.notna()].dt.tz_convert("Asia/Kolkata")
    df["hour"] = ist.dt.hour.astype(int)
    df["dow"] = ist.dt.dayofweek.astype(int)   # 0=Mon
    df["date"] = ist.dt.date

    # violations + severity
    df["vlist"] = df["violation_type"].map(parse_violations)
    df["S"] = df["vlist"].map(severity_of)

    # footprint
    df["F"] = df["vehicle_type"].map(lambda v: FOOTPRINT.get(v, DEFAULT_FOOTPRINT))

    # junction amplifier
    df["at_junction"] = (df["junction_name"].notna() &
                         (df["junction_name"] != "No Junction"))
    df["J"] = np.where(df["at_junction"], 1.6, 1.0)

    # temporal coupling with baseline traffic demand (IST clock)
    def t_weight(h):
        if 8 <= h <= 11 or 17 <= h <= 21:   # AM / PM rush windows
            return 1.4
        if 7 == h or 12 <= h <= 16:         # shoulder / daytime
            return 1.1
        return 0.8                          # night / off-peak
    df["T"] = df["hour"].map(t_weight)

    print(f"  rows after clean: {len(df):,}")
    return df


# --------------------------------------------------------------------------- #
# 2. Chronicity (repeat offenders + persistent locations)
# --------------------------------------------------------------------------- #
def add_chronicity(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing chronicity ...")
    # repeat-offender factor per vehicle (log-scaled, capped)
    vc = df["vehicle_number"].value_counts()
    df["veh_repeat"] = df["vehicle_number"].map(vc)
    # location persistence: count per ~220m cell (computed after gridding below)
    df["C"] = np.clip(np.log1p(df["veh_repeat"] - 1) * 0.15, 0, 0.6)
    return df


# --------------------------------------------------------------------------- #
# 3. PCII + spatial grid
# --------------------------------------------------------------------------- #
def compute_pcii(df: pd.DataFrame) -> pd.DataFrame:
    df["PCII"] = BASE * df["S"] * df["F"] * df["J"] * df["T"] * (1 + df["C"])
    return df


def gridify(df: pd.DataFrame):
    df["gx"] = np.floor(df["longitude"] / CELL_DEG).astype(int)
    df["gy"] = np.floor(df["latitude"] / CELL_DEG).astype(int)
    df["cell"] = list(zip(df["gx"], df["gy"]))
    return df


# --------------------------------------------------------------------------- #
# 4. Getis-Ord Gi* hotspot statistic
# --------------------------------------------------------------------------- #
def getis_ord(cell_value: dict[tuple, float], radius: int = GISTAR_RADIUS):
    """Return {cell: z_score}. Binary distance-band weights (incl. self)."""
    cells = list(cell_value.keys())
    x = np.array([cell_value[c] for c in cells], dtype=float)
    n = len(cells)
    if n < 3:
        return {c: 0.0 for c in cells}
    xbar = x.mean()
    S = math.sqrt((x ** 2).mean() - xbar ** 2) or 1e-9

    idx = {c: i for i, c in enumerate(cells)}
    z = {}
    for c in cells:
        gx, gy = c
        neigh = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nb = (gx + dx, gy + dy)
                if nb in idx:
                    neigh.append(idx[nb])
        W = len(neigh)                       # binary weights -> sum == count
        lag = x[neigh].sum()
        denom = S * math.sqrt((n * W - W * W) / (n - 1))
        z[c] = (lag - xbar * W) / denom if denom else 0.0
    return z


# --------------------------------------------------------------------------- #
# 5. Patrol optimizer (greedy spatial max-coverage)
# --------------------------------------------------------------------------- #
def optimize_patrols(hotspots: list[dict], k: int, sep_cells: int = 4):
    """Pick k hotspots maximizing total impact with spatial separation
    (avoid stacking patrols on the same cluster)."""
    chosen = []
    used = []  # list of (gx, gy)
    for h in sorted(hotspots, key=lambda d: -d["pcii"]):
        if len(chosen) >= k:
            break
        gx, gy = h["gx"], h["gy"]
        if all(max(abs(gx - ux), abs(gy - uy)) > sep_cells for ux, uy in used):
            chosen.append(h)
            used.append((gx, gy))
    return chosen


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cell_center(gx, gy):
    return (gy + 0.5) * CELL_DEG, (gx + 0.5) * CELL_DEG  # lat, lon


def best_label(sub: pd.DataFrame) -> str:
    """Human-readable name for a hotspot cell."""
    j = sub.loc[sub["at_junction"], "junction_name"]
    if len(j):
        name = j.mode().iloc[0]
        return re.sub(r"^BTP\d+\s*-\s*", "", name)
    ps = sub["police_station"].mode()
    ps = ps.iloc[0] if len(ps) else "Unknown"
    # try a short locality from the address
    loc = sub["location"].dropna()
    if len(loc):
        first = loc.iloc[0].split(",")
        seg = first[0].strip() if first else ""
        if seg and not seg[0].isdigit():
            return f"{seg} ({ps})"
    return ps


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    df = load()
    df = add_chronicity(df)
    df = compute_pcii(df)
    df = gridify(df)

    print("Aggregating grid ...")
    g = df.groupby("cell")
    cell_pcii = g["PCII"].sum().to_dict()
    cell_count = g.size().to_dict()

    print("Running Getis-Ord Gi* ...")
    zscore = getis_ord(cell_pcii)

    # ---- build hotspot records ----
    print("Building hotspots ...")
    groups = {cell: sub for cell, sub in df.groupby("cell")}
    hotspots = []
    for cell, pcii in cell_pcii.items():
        gx, gy = cell
        lat, lon = cell_center(gx, gy)
        sub = groups[cell]
        cnt = int(cell_count[cell])
        vt_counts = Counter(v for lst in sub["vlist"] for v in lst)
        hotspots.append({
            "gx": gx, "gy": gy, "lat": round(lat, 6), "lon": round(lon, 6),
            "pcii": round(float(pcii), 1),
            "count": cnt,
            "z": round(float(zscore.get(cell, 0)), 2),
            "intensity_per_violation": round(float(pcii) / cnt, 2),
            "junction_share": round(float(sub["at_junction"].mean()), 2),
            "name": best_label(sub),
            "top_violations": [v for v, _ in vt_counts.most_common(3)],
            "top_vehicle": sub["vehicle_type"].mode().iloc[0] if len(sub) else "",
            "repeat_offenders": int((sub["veh_repeat"] >= 3).sum()),
        })

    hotspots.sort(key=lambda d: -d["pcii"])

    # significance class for every spot (Gi* z thresholds)
    for h in hotspots:
        h["significance"] = (
            "Very High" if h["z"] >= 2.58 else
            "High" if h["z"] >= 1.96 else
            "Moderate" if h["z"] >= 1.65 else "Emerging"
        )
    top = hotspots[:TOP_HOTSPOTS]

    # ---- patrol plan (optimize over ALL hotspots, not just the top slice) ----
    print("Optimizing patrols ...")
    plan = optimize_patrols(hotspots, N_PATROLS)

    # ---- temporal profiles for top hotspots + patrol cells (24x7) ----
    print("Temporal profiles ...")
    profile_cells = {(h["gx"], h["gy"]) for h in top} | {(h["gx"], h["gy"]) for h in plan}
    tdf = df[df["cell"].isin(profile_cells)]
    profiles = {}
    for cell, sub in tdf.groupby("cell"):
        mat = np.zeros((7, 24))
        agg = sub.groupby(["dow", "hour"])["PCII"].sum()
        for (d, h), v in agg.items():
            mat[int(d), int(h)] = v
        profiles[f"{cell[0]}_{cell[1]}"] = [[round(v, 1) for v in row] for row in mat]

    # recommended deployment window per patrol unit
    for rank, h in enumerate(plan, 1):
        key = f"{h['gx']}_{h['gy']}"
        prof = np.array(profiles.get(key, np.zeros((7, 24))))
        hourly = prof.sum(axis=0)
        peak_h = int(hourly.argmax()) if hourly.sum() else 9
        h["patrol_rank"] = rank
        h["peak_hour"] = peak_h
        h["peak_window"] = f"{peak_h:02d}:00\u2013{(peak_h+2)%24:02d}:00"

    # ---- city KPIs ----
    total_pcii = float(df["PCII"].sum())
    sig_spots = [h for h in hotspots if h["z"] >= 1.96]
    # share of total impact concentrated in top 1% of cells (Pareto signal)
    sorted_p = np.sort([h["pcii"] for h in hotspots])[::-1]
    top1pct_n = max(1, len(sorted_p) // 100)
    pareto = float(sorted_p[:top1pct_n].sum() / total_pcii * 100)

    kpis = {
        "total_records": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "total_pcii": round(total_pcii, 0),
        "n_cells": len(hotspots),
        "n_significant_hotspots": len(sig_spots),
        "pareto_top1pct_share": round(pareto, 1),
        "patrols": len(plan),
        "patrol_pcii_covered": round(sum(h["pcii"] for h in plan), 0),
        "patrol_pcii_share": round(sum(h["pcii"] for h in plan) / total_pcii * 100, 1),
        "cell_size_m": int(CELL_DEG * 111000),
    }

    # ---- aux breakdowns ----
    veh_break = df.groupby("vehicle_type")["PCII"].sum().sort_values(ascending=False)
    vio_break = Counter()
    for lst, p in zip(df["vlist"], df["PCII"]):
        for v in lst:
            vio_break[v] += p
    citywide_hourly = df.groupby("hour")["PCII"].sum().reindex(range(24), fill_value=0)
    citywide_dow = df.groupby("dow")["PCII"].sum().reindex(range(7), fill_value=0)
    ps_break = df.groupby("police_station")["PCII"].sum().sort_values(ascending=False)

    # heatmap points (all populated cells, lightweight)
    heat = [[h["lat"], h["lon"], round(h["pcii"], 1)] for h in hotspots]

    # ---- write outputs ----
    print("Writing JSON ...")
    dump = lambda name, obj: json.dump(
        obj, open(os.path.join(OUT, name), "w", encoding="utf-8"))

    dump("kpis.json", kpis)
    dump("hotspots.json", top)
    dump("heat.json", heat)
    dump("profiles.json", profiles)
    dump("patrol_plan.json", plan)
    dump("breakdowns.json", {
        "vehicle": [{"k": k, "v": round(float(v), 0)} for k, v in veh_break.head(12).items()],
        "violation": [{"k": k, "v": round(float(v), 0)} for k, v in
                      sorted(vio_break.items(), key=lambda x: -x[1])[:12]],
        "hourly": [round(float(v), 0) for v in citywide_hourly],
        "dow": [round(float(v), 0) for v in citywide_dow],
        "police_station": [{"k": k, "v": round(float(v), 0)} for k, v in ps_break.head(12).items()],
    })

    # also bundle everything into a single JS file so the dashboard opens
    # with a plain double-click (no local web server / fetch needed).
    bundle = {
        "kpis": kpis,
        "hotspots": top,
        "heat": heat,
        "profiles": profiles,
        "patrol_plan": plan,
        "breakdowns": {
            "vehicle": [{"k": k, "v": round(float(v), 0)} for k, v in veh_break.head(12).items()],
            "violation": [{"k": k, "v": round(float(v), 0)} for k, v in
                          sorted(vio_break.items(), key=lambda x: -x[1])[:12]],
            "hourly": [round(float(v), 0) for v in citywide_hourly],
            "dow": [round(float(v), 0) for v in citywide_dow],
            "police_station": [{"k": k, "v": round(float(v), 0)} for k, v in ps_break.head(12).items()],
        },
    }
    dash_dir = os.path.join(HERE, "dashboard")
    os.makedirs(dash_dir, exist_ok=True)
    with open(os.path.join(dash_dir, "data.js"), "w", encoding="utf-8") as fh:
        fh.write("window.GRID = ")
        json.dump(bundle, fh)
        fh.write(";")

    print("\n=== DONE ===")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
    print(f"\n  Outputs in: {OUT}")


if __name__ == "__main__":
    main()
