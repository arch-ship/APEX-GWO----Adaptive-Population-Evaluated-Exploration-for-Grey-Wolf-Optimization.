# APEX-GWO: Adaptive Population-Evaluated Exploration for Grey Wolf Optimization

**APEX-GWO** is a modular Grey Wolf Optimization (GWO) ablation platform and 12-problem benchmarking suite built with **FastAPI**, **NumPy**, and a self-contained **HTML5 Canvas** frontend deployed on **Vercel**.

Standard Grey Wolf Optimization relies on a strictly linear convergence factor ($a(t) = 2(1 - t/(T-1))$) that enforces a fixed 50/50 exploration-exploitation split and selects initial leaders using scalar objective fitness alone. APEX-GWO isolates and evaluates two parameter-free/lightweight improvements through a structured $2 \times 2$ ablation study across 6 maximization and 6 minimization problems.

---

## Table of Contents

- [4-Variant Ablation Matrix](#4-variant-ablation-matrix)
- [Core Algorithmic Upgrades](#core-algorithmic-upgrades)
- [12-Problem Benchmark Suite](#12-problem-benchmark-suite)
- [Verified Benchmark Results](#verified-benchmark-results)
- [Repository Structure](#repository-structure)
- [API Endpoints](#api-endpoints)
- [Running Locally](#running-locally)
- [Deploying on Vercel](#deploying-on-vercel)
- [Contributors & Course Info](#contributors--course-info)

---

## 4-Variant Ablation Matrix

Each variant lives in its own standalone, comment-free Python module under `optimizers/` and can be inspected directly from the web dashboard via the **Show Code** button.

| Code | Algorithm Name | Convergence Factor $a(t)$ | Leader Selection ($\alpha, \beta, \delta$) | Module |
| :--- | :--- | :--- | :--- | :--- |
| **GWO-01** | Original GWO | Linear: $2\left(1 - \frac{t}{T-1}\right)$ | Fitness Only | `optimizers/gwo_original.py` |
| **GWO-02** | Non-Linear $a$ GWO | Cosine: $2\cos\!\left(\frac{\pi}{2}\left(\frac{t}{T-1}\right)^{1.25}\right)$ | Fitness Only | `optimizers/gwo_nonlinear_a.py` |
| **GWO-03** | Dual-Criteria GWO | Linear: $2\left(1 - \frac{t}{T-1}\right)$ | Step-1 Top-3 Column Tiers ($t = 0$) | `optimizers/gwo_dual_criteria.py` |
| **GWO-04** | Combined APEX-GWO | Cosine: $2\cos\!\left(\frac{\pi}{2}\left(\frac{t}{T-1}\right)^{1.25}\right)$ | Step-1 Top-3 Column Tiers ($t = 0$) | `optimizers/gwo_hybrid.py` |

---

## Core Algorithmic Upgrades

1. **Non-Linear Cosine Convergence Factor (`GWO-02`, `GWO-04`)**
   $$a(t) = 2\cos\!\left(\frac{\pi}{2}\left(\frac{t}{T - 1}\right)^{1.25}\right)$$
   Sustains $a(t) > 1.0$ across approximately **70%** of the iteration budget for wider global exploration on multimodal landscapes before transitioning sharply into local exploitation during the final **30%**.

2. **Step-1 Top-3 Tiered Column-Ranking Leader Selection (`GWO-03`, `GWO-04`)**
   At initialization ($t = 0$), the pack coordinates and objective scores are stacked into an $N \times (D + 1)$ matrix $[x_1, \dots, x_D, f(x)]$. Each column is ranked independently (ascending for minimization, descending for maximization) to tag its 1st (`Rank-1`), 2nd (`Rank-2`), and 3rd (`Rank-3`) best values:
   - $\alpha$ is assigned to the wolf with the highest `Rank-1` count.
   - $\beta$ is assigned from the remaining wolves with the highest `Rank-2` count.
   - $\delta$ is assigned from the remaining wolves with the highest `Rank-3` count.
   - **Whole-row integrity** is strictly preserved (coordinates are never spliced across wolves), and standard fitness updates take over for $t \ge 1$ to prevent boundary corner drift on non-monotonic interior optima.

3. **Fair Tie Exclusion**
   Winner evaluation checks rounded scores within a tolerance of $10^{-4}$. Exact ties are never credited to any algorithm's win count; instead, they are counted separately in a **Ties (Unassigned)** tile and listed explicitly in a dedicated **Tied Algorithms** column (both in the dashboard and CSV exports).

---

## 12-Problem Benchmark Suite

All four variants support minimization and maximization natively via a shared `is_better(candidate, reference, mode)` comparator and identical seeded initial populations.

| # | Problem | Category | Mode | Dim | Encoding / Constraint Handling |
| :-: | :--- | :--- | :-: | :-: | :--- |
| 1 | **0/1 Knapsack (20 Items)** | Combinatorial | MAX | 20D | Sigmoid binary transfer; quadratic weight penalty |
| 2 | **Traveling Salesman (14 Cities)** | Routing | MIN | 14D | Random-Key `argsort` permutation; closed tour |
| 3 | **Max Network Flow (7 Nodes)** | Graph Flow | MAX | 11D | Edge capacity bounds; node flow conservation penalty |
| 4 | **Cloud VM Task Scheduling** | Scheduling | MIN | 20D | Discrete task-to-VM mapping; makespan + energy |
| 5 | **Warehouse Bin Packing** | Packing | MIN | 18D | Random-Key First-Fit bin count + slack ratio |
| 6 | **Portfolio Sharpe Optimization** | Finance | MAX | 8D | Simplex weights; 45% single-asset concentration cap |
| 7 | **Facility Location Coverage** | Logistics | MAX | 6D | Continuous 2D hub coordinates; 24 km demand radius |
| 8 | **Weighted Graph Max-Cut** | Graph Partition | MAX | 16D | Sign-threshold binary partition; total cut weight |
| 9 | **Hybrid IR Search Rank Tuner** | Info Retrieval | MAX | 5D | BM25 ($k_1, b$) + semantic/trust fusion; 40 ms cap |
| 10 | **Rastrigin Multimodal** | Continuous | MIN | 10D | Dense cosine local optima trap on $[-5.12, 5.12]^{10}$ |
| 11 | **Ackley Deep Funnel** | Continuous | MIN | 10D | Flat outer basin + sharp origin funnel on $[-10, 10]^{10}$ |
| 12 | **Rosenbrock Valley** | Continuous | MIN | 10D | Curved non-convex valley with optimum at $(1, \dots, 1)$ |

---

## Verified Benchmark Results

Results below were recorded at the default configuration ($N = 20$, $T = 40$, $\text{seed} = 42$) with strict tie exclusion:

| Algorithm | Outright Wins | Avg $\Delta$ vs. Original |
| :--- | :---: | :---: |
| **Combined APEX-GWO (GWO-04)** | **5 / 12** | **+13.99%** |
| **Dual-Criteria GWO (GWO-03)** | 2 / 12 | — |
| **Original GWO (GWO-01)** | 1 / 12 | Baseline |
| **Non-Linear $a$ GWO (GWO-02)** | 0 / 12 *(shares 3 ties)* | +7.37% |
| **Ties (Excluded from Wins)** | 4 / 12 | Knapsack, Bin Packing, Max-Cut, Hybrid IR |

- **Statistical Validation (10-Trial Wilcoxon Signed-Rank Test, seeds 42 to 195):** On 10D Ackley, Combined APEX-GWO reduces mean error by **95%** ($0.2102 \to 0.0105$) and variance by over $70\times$ compared with Original GWO ($p = 0.0020 < 0.01$).

---

## Repository Structure

```text
.
├── environments/
│   ├── __init__.py
│   └── problem_suite.py          # 12 Min/Max benchmark problems + 2D surfaces
├── optimizers/
│   ├── __init__.py               # Algorithm registry (ALGORITHMS, ALGO_ORDER)
│   ├── common.py                 # RunResult, is_better, pack_diversity, convergence_iteration
│   ├── gwo_original.py           # GWO-01: Original GWO (Linear a, Fitness Only)
│   ├── gwo_nonlinear_a.py        # GWO-02: Non-Linear Cosine a GWO
│   ├── gwo_dual_criteria.py      # GWO-03: Step-1 Top-3 Column-Tier GWO
│   └── gwo_hybrid.py             # GWO-04: Combined APEX-GWO
├── index.html                    # Self-contained UI (HTML + CSS + JS + Canvas charts)
├── main.py                       # FastAPI backend entrypoint & tie-excluded scoring
├── pyproject.toml                # Python >=3.12 project config + Vercel entrypoint
├── requirements.txt              # FastAPI, Uvicorn, NumPy
├── vercel.json                   # Serverless function timeout config (60s)
├── .gitignore
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the self-contained web dashboard (`index.html`) |
| `GET` | `/api/health` | Health check returning status, algorithm count, and frontend check |
| `GET` | `/api/meta` | Returns metadata for all 4 algorithms, 12 problems, and 2D surfaces |
| `GET` | `/api/code/{algo_id}` | Returns the clean Python source code for a single algorithm |
| `POST` | `/api/run` | Runs all 4 variants on a single problem and returns curves, audit, and decoded solution |
| `POST` | `/api/suite` | Executes the full 12-problem benchmark suite with tie-excluded scoreboard |
| `POST` | `/api/arena2d` | Computes 2D surface grid and per-iteration trajectories for the live 2D Swarm View |

---

## Running Locally

### Prerequisites
- Python `3.12+`

### Setup & Launch

```bash
# 1. Clone the repository
git clone https://github.com/arch-ship/APEX-GWO----Adaptive-Population-Evaluated-Exploration-for-Grey-Wolf-Optimization..git
cd APEX-GWO----Adaptive-Population-Evaluated-Exploration-for-Grey-Wolf-Optimization.

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server
uvicorn main:app --reload --port 8000
```

Open **`http://localhost:8000`** in your browser (or verify **`http://localhost:8000/api/health`**).

---

## Deploying on Vercel

This repository is pre-configured for zero-config FastAPI deployment on Vercel via `pyproject.toml` (`entrypoint = "main:app"`) and `vercel.json`.

1. Import the GitHub repository into **Vercel**.
2. In **Project Settings -> Build and Deployment**:
   - **Framework Preset:** `FastAPI`
   - **Root Directory:** leave **blank** (repository root)
   - **Build Command / Output Directory overrides:** leave **OFF**
3. Deploy and verify `https://<your-vercel-domain>/api/health` returns:
   ```json
   {"status": "ok", "algorithms": 4, "index_found": true}
   ```

---

## Contributors & Course Info

- **Institution:** Jaypee Institute of Information Technology (JIIT), Sector 62, Noida
- **Course:** Introduction to Meta Heuristic Optimization (Project-Based Learning)
- **Authors:** Aditya Garg, Ayaz Wasee Al Ansari
- **Faculty Guide:** Dr. Parul Agarwal
