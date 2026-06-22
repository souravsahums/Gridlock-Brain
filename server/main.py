"""
GridLock Brain API
==================
FastAPI service that serves the dashboard from a live PostgreSQL/PostGIS database.

Key trick: GET /data.js reconstructs the exact `window.GRID = {...}` bundle that the
existing dashboard expects — but built FROM THE DATABASE — so the front-end is
unchanged yet fully DB-driven.
"""
import json
import os
from fastapi import FastAPI, Response, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from db import get_conn

app = FastAPI(title="GridLock Brain API", version="1.0")

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

SIG_RANK = {"Emerging": 0, "Moderate": 1, "High": 2, "Very High": 3}


# --------------------------------------------------------------------------- #
def build_bundle():
    """Reconstruct the dashboard bundle from relational tables."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT props FROM hotspots ORDER BY pcii DESC")
        hotspots = [r["props"] for r in cur.fetchall()]
        cur.execute("SELECT props FROM hotspots WHERE patrol_rank IS NOT NULL "
                    "ORDER BY patrol_rank")
        patrol_plan = [r["props"] for r in cur.fetchall()]
        cur.execute("SELECT key, value FROM app_data")
        kv = {r["key"]: r["value"] for r in cur.fetchall()}
    return {
        "kpis": kv.get("kpis", {}),
        "hotspots": hotspots,
        "heat": kv.get("heat", []),
        "profiles": kv.get("profiles", {}),
        "patrol_plan": patrol_plan,
        "breakdowns": kv.get("breakdowns", {}),
        "validation": kv.get("validation", {}),
    }


# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM violations")
            v = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM hotspots")
            h = cur.fetchone()["n"]
        return {"status": "ok", "violations": v, "hotspots": h}
    except Exception as e:                       # noqa: BLE001
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/api/bundle")
def bundle():
    return build_bundle()


@app.get("/data.js")
def data_js():
    js = "window.GRID = " + json.dumps(build_bundle()) + ";"
    return Response(js, media_type="application/javascript")


@app.get("/api/hotspots")
def hotspots(min_significance: str = Query("Emerging"), limit: int = 100):
    floor = SIG_RANK.get(min_significance, 0)
    keep = [k for k, v in SIG_RANK.items() if v >= floor]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT cell_id, name, lat, lon, pcii, gistar_z, significance, "
            "ticket_count, patrol_rank, peak_window "
            "FROM hotspots WHERE significance = ANY(%s) "
            "ORDER BY pcii DESC LIMIT %s", (keep, limit))
        return cur.fetchall()


@app.get("/api/hotspots/near")
def hotspots_near(lat: float, lon: float, km: float = 2.0, limit: int = 20):
    """PostGIS spatial query — hotspots within `km` of a point (e.g. a patrol car)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT cell_id, name, pcii, significance, peak_window, "
            "ROUND((ST_Distance(geom::geography, "
            "       ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography)/1000.0)::numeric,2) AS km "
            "FROM hotspots "
            "WHERE ST_DWithin(geom::geography, "
            "      ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography, %s) "
            "ORDER BY pcii DESC LIMIT %s",
            (lon, lat, lon, lat, km * 1000, limit))
        return cur.fetchall()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


# static assets (after routes so /data.js wins)
app.mount("/", StaticFiles(directory=STATIC), name="static")
