"""
ETL: load raw violations + computed intelligence into PostgreSQL/PostGIS.

Run locally against the provisioned Postgres public endpoint:

    set DATABASE_URL=postgresql://user:pwd@host:5432/gridlock?sslmode=require
    python etl.py

Reads:
  - ../jan to may police violation_anonymized791b166.csv   (raw violations)
  - ../dashboard/data.js                                   (computed bundle)
"""
import io
import json
import os

import numpy as np
import pandas as pd
import psycopg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")           # repo root (solution/)
DATASET_NAME = "jan to may police violation_anonymized791b166.csv"


def find_csv():
    candidates = [
        os.environ.get("GRIDLOCK_CSV"),
        os.path.join(ROOT, "data", "violations.csv"),
        os.path.join(ROOT, "data", DATASET_NAME),
        os.path.join(ROOT, DATASET_NAME),
        os.path.join(ROOT, "..", DATASET_NAME),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Dataset CSV not found. Set GRIDLOCK_CSV or place it at data/violations.csv.")


DATA_JS = os.path.join(HERE, "..", "dashboard", "data.js")
DATABASE_URL = os.environ["DATABASE_URL"]

DDL = """
CREATE EXTENSION IF NOT EXISTS postgis;

DROP TABLE IF EXISTS violations;
CREATE TABLE violations (
    id               text,
    latitude         double precision,
    longitude        double precision,
    location         text,
    vehicle_number   text,
    vehicle_type     text,
    violation_type   text,
    created_datetime timestamptz,
    police_station   text,
    junction_name    text
);

DROP TABLE IF EXISTS hotspots;
CREATE TABLE hotspots (
    cell_id      text PRIMARY KEY,
    name         text,
    lat          double precision,
    lon          double precision,
    geom         geometry(Point, 4326),
    pcii         double precision,
    gistar_z     double precision,
    significance text,
    ticket_count int,
    patrol_rank  int,
    peak_window  text,
    props        jsonb
);

DROP TABLE IF EXISTS app_data;
CREATE TABLE app_data (key text PRIMARY KEY, value jsonb);
"""


def load_bundle():
    raw = open(DATA_JS, encoding="utf-8").read()
    return json.loads(raw[len("window.GRID = "):-1])


def main():
    print("Reading CSV ...")
    df = pd.read_csv(find_csv(), low_memory=False, usecols=[
        "id", "latitude", "longitude", "location", "vehicle_number",
        "vehicle_type", "violation_type", "created_datetime",
        "police_station", "junction_name"])
    df = df[(df.latitude.between(12.7, 13.4)) & (df.longitude.between(77.3, 77.9))]
    print(f"  rows: {len(df):,}")

    bundle = load_bundle()

    print("Connecting + creating schema ...")
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(DDL)

        # ---- bulk load raw violations via COPY ----
        print("COPY violations ...")
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)
        with cur.copy(
            "COPY violations (id,latitude,longitude,location,vehicle_number,"
            "vehicle_type,violation_type,created_datetime,police_station,"
            "junction_name) FROM STDIN WITH (FORMAT csv)"
        ) as cp:
            cp.write(buf.read())

        # ---- hotspots (merge top-list + patrol plan) ----
        print("Inserting hotspots ...")
        merged = {}
        for h in bundle["hotspots"]:
            merged[f"{h['gx']}_{h['gy']}"] = dict(h)
        for p in bundle["patrol_plan"]:
            key = f"{p['gx']}_{p['gy']}"
            base = merged.get(key, dict(p))
            base["patrol_rank"] = p.get("patrol_rank")
            base["peak_window"] = p.get("peak_window")
            base["peak_hour"] = p.get("peak_hour")
            merged[key] = base

        rows = []
        for key, h in merged.items():
            rows.append((
                key, h.get("name"), h["lat"], h["lon"], h["lon"], h["lat"],
                h["pcii"], h.get("z"), h.get("significance"), h.get("count"),
                h.get("patrol_rank"), h.get("peak_window"), json.dumps(h)))
        cur.executemany(
            "INSERT INTO hotspots (cell_id,name,lat,lon,geom,pcii,gistar_z,"
            "significance,ticket_count,patrol_rank,peak_window,props) VALUES "
            "(%s,%s,%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326),%s,%s,%s,%s,%s,%s,%s)",
            rows)
        cur.execute("CREATE INDEX hotspots_geom_idx ON hotspots USING GIST (geom)")
        cur.execute("CREATE INDEX violations_created_idx ON violations (created_datetime)")

        # ---- app_data key/value ----
        print("Inserting app_data ...")
        for key in ("kpis", "heat", "profiles", "breakdowns", "validation"):
            cur.execute("INSERT INTO app_data (key,value) VALUES (%s,%s) "
                        "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                        (key, json.dumps(bundle.get(key, {}))))

        cur.execute("SELECT count(*) FROM violations")
        nv = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM hotspots")
        nh = cur.fetchone()[0]
    print(f"\nDONE. violations={nv:,}  hotspots={nh}")


if __name__ == "__main__":
    main()
