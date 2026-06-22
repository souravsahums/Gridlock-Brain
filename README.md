# 🧠 GridLock Brain
### AI-Driven Parking Intelligence for Targeted Congestion Enforcement

> Detect illegal-parking hotspots and **quantify their impact on traffic flow — without a single traffic sensor** — then turn that into an optimized, time-aware patrol plan.
>
> Built on ~298,000 anonymized parking-violation records.

### 🔗 Live demo
- **Interactive dashboard:** https://gridlock-api.icygrass-96b34d95.centralus.azurecontainerapps.io
- **Video walkthrough:** https://drive.google.com/file/d/1CtR2cVCyUZ-XJRkfcJwmV5uTlN_nKLIr/view?usp=sharing

> The live dashboard is served from a PostgreSQL/PostGIS database via a REST API. You can also run **everything locally with zero cloud resources** — see [Run locally](#-run-locally).

---

## 🚀 The three core ideas

### 1. PCII — Parking Congestion Impact Index
A raw ticket count tells you *where a fine was written*, not *where traffic actually suffers*. PCII converts every violation into an estimate of **carriageway disruption** by fusing five signals already present in the data:

| Factor | Meaning | Source |
|---|---|---|
| **S** — Severity | how much the violation blocks a lane (double-parking & main-road ≫ no-parking) | `violation_type` |
| **F** — Footprint | road length the vehicle occupies (a lorry ≈ 6× a scooter) | `vehicle_type` |
| **J** — Junction amplifier | blocking near an intersection cascades upstream | `junction_name` |
| **T** — Temporal coupling | a car parked at 9 AM hurts more than at 3 AM | `created_datetime` |
| **C** — Chronicity | repeat-offender / persistent black-spot amplifier | `vehicle_number` |

```
PCII = BASE · S · F · J · T · (1 + C)
```

No external traffic feed required — so it deploys to any city with ticketing data.

### 2. Statistically-significant hotspots (Getis-Ord Gi\*)
A spatial statistic on a ~220 m grid separates **real choke-clusters** from random noise. Each hotspot gets a confidence class tied to standard z-score significance levels:

| Class | z-score | Confidence |
|---|---|---|
| Very High | ≥ 2.58 | 99% |
| High | ≥ 1.96 | 95% |
| Moderate | ≥ 1.65 | 90% |
| Emerging | < 1.65 | below 90% |

### 3. Optimized, time-aware patrol deployment
A greedy max-coverage optimizer with spatial separation allocates a fixed fleet for maximum congestion relief — and gives each unit a recommended shift window from its own 24×7 demand profile.

> **Headline:** the top **1%** of locations carry **35%** of all congestion impact, and **25** well-placed patrols cover **~18%** of the entire city's impact.

---

## ✅ The model is validated, not arbitrary
Validation uses **only the provided dataset** ([validate.py](validate.py)):

| Test | Result | Meaning |
|---|---|---|
| Predictive (70/30 temporal hold-out) | Spearman **0.70**, precision@50 **78%** | hotspots predict *future* violations |
| Robustness (±25% weight perturbation, 100×) | top-50 **96% stable** | conclusions don't depend on exact weights |
| Convergent (vs. chronic active-days) | **0.91** | high-PCII places are chronically obstructed |
| Spatial autocorrelation (Moran's I) | **0.64**, p=**0.005** | impact is significantly clustered |

---

## 🖥️ Run locally

The repository ships with a pre-computed `dashboard/data.js`, so you can see the full dashboard **without any dataset or cloud setup**.

### Option A — just view the dashboard (fastest)
Open [`dashboard/index.html`](dashboard/index.html) in any modern browser (double-click it).
*(An internet connection is used only for the map tiles.)*

### Option B — regenerate everything from the raw data
1. **Get the dataset** (the anonymized parking-violation CSV — not redistributed in this repo) and put it at:
   ```
   data/violations.csv
   ```
   *(or point to it with an environment variable: `set GRIDLOCK_CSV=C:\path\to\violations.csv`)*
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline** (computes PCII, Gi\* hotspots, temporal profiles, patrol plan, and regenerates `dashboard/data.js`):
   ```bash
   python pipeline.py
   ```
4. **Run the validation suite** (optional — injects trust metrics into the dashboard):
   ```bash
   python validate.py
   ```
5. **Open the dashboard:** [`dashboard/index.html`](dashboard/index.html)

On Windows you can do steps 2–5 in one click with [`run.bat`](run.bat).

---

## 🗄️ Optional: containerized API (bring your own database)
The [`server/`](server) folder contains a FastAPI service that serves the dashboard from a PostgreSQL/PostGIS database, making it a deployable service instead of a static page:
- [`server/main.py`](server/main.py) — REST API (`/api/health`, `/api/hotspots`, `/api/hotspots/near` PostGIS spatial query, `/data.js`).
- [`server/etl.py`](server/etl.py) — loads the CSV + computed intelligence into Postgres.
- [`Dockerfile`](Dockerfile) — builds the API image.

Set `DATABASE_URL` to your own Postgres instance, run `python server/etl.py` to load it, then run the container. No cloud account is required to run it against a local Postgres.

---

## 📁 Project structure
```
.
├── pipeline.py          # PCII + Getis-Ord Gi* + temporal profiles + patrol optimizer
├── validate.py          # predictive / robustness / spatial validation suite
├── run.bat              # one-click: pipeline -> validate -> open dashboard
├── requirements.txt
├── dashboard/
│   ├── index.html       # interactive command-center dashboard
│   └── data.js          # pre-computed demo data (so it runs out-of-the-box)
└── server/              # optional FastAPI + PostGIS service
    ├── main.py
    ├── db.py
    ├── etl.py
    └── requirements.txt
```

---

## 🔭 Roadmap
- Calibrate PCII against ground-truth speeds to report impact in **minutes of delay** and **₹ cost**.
- Live ingestion + auto-alerting on **emerging** hotspots.
- Reinforcement-learning patrol routing that learns deterrence decay.
- ANPR camera fusion for sensor-light coverage.

---

## 🛠️ Tech stack
Python (pandas / numpy) · Getis-Ord Gi\* spatial statistics · Monte-Carlo robustness testing · Leaflet dashboard · FastAPI + PostgreSQL/PostGIS.

## 📄 License
MIT — see [LICENSE](LICENSE).

Made by **Sourav Sahu** — *See the gridlock before it forms.*
