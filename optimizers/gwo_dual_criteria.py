import time
import numpy as np
from optimizers.common import RunResult, is_better, pack_diversity, convergence_iteration


def select_tiered_column_leaders(positions, fitness_vals, mode="min"):
    table = np.column_stack([positions, fitness_vals])
    n_wolves, n_cols = table.shape
    tier1 = np.zeros(n_wolves, dtype=int)
    tier2 = np.zeros(n_wolves, dtype=int)
    tier3 = np.zeros(n_wolves, dtype=int)

    for c in range(n_cols):
        col = table[:, c]
        uniq = np.unique(np.round(col, 12))
        ordered_vals = uniq if mode == "min" else uniq[::-1]
        if len(ordered_vals) >= 1:
            tier1 += np.isclose(col, ordered_vals[0], rtol=0.0, atol=1e-12).astype(int)
        if len(ordered_vals) >= 2:
            tier2 += np.isclose(col, ordered_vals[1], rtol=0.0, atol=1e-12).astype(int)
        if len(ordered_vals) >= 3:
            tier3 += np.isclose(col, ordered_vals[2], rtol=0.0, atol=1e-12).astype(int)

    remaining = list(range(n_wolves))

    def fit_key(idx):
        return -fitness_vals[idx] if mode == "min" else fitness_vals[idx]

    a_idx = max(remaining, key=lambda i: (int(tier1[i]), int(tier2[i]), int(tier3[i]), fit_key(i)))
    remaining.remove(a_idx)

    if remaining:
        b_idx = max(remaining, key=lambda i: (int(tier2[i]), int(tier1[i] + tier2[i]), int(tier3[i]), fit_key(i)))
        remaining.remove(b_idx)
    else:
        b_idx = a_idx

    if remaining:
        d_idx = max(remaining, key=lambda i: (int(tier3[i]), int(tier1[i] + tier2[i] + tier3[i]), fit_key(i)))
    else:
        d_idx = b_idx

    records = [
        {
            "index": i,
            "t1": int(tier1[i]),
            "t2": int(tier2[i]),
            "t3": int(tier3[i]),
            "total_cols": n_cols,
        }
        for i in range(n_wolves)
    ]
    return a_idx, b_idx, d_idx, records


def run_dual_criteria_gwo(
    obj_func,
    lb,
    ub,
    mode="min",
    n_agents=20,
    max_iter=40,
    seed=42,
    record_2d=False,
    **kwargs,
):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = len(lb)

    positions = lb + rng.random((n_agents, dim)) * (ub - lb)
    fitness_vals = np.array([float(obj_func(p)) for p in positions], dtype=float)

    a_idx, b_idx, d_idx, records = select_tiered_column_leaders(positions, fitness_vals, mode)
    alpha_pos, alpha_score = positions[a_idx].copy(), float(fitness_vals[a_idx])
    beta_pos, beta_score = positions[b_idx].copy(), float(fitness_vals[b_idx])
    delta_pos, delta_score = positions[d_idx].copy(), float(fitness_vals[d_idx])

    best_idx = int(np.argmin(fitness_vals) if mode == "min" else np.argmax(fitness_vals))
    best_pos = positions[best_idx].copy()
    best_fit = float(fitness_vals[best_idx])

    roles = {a_idx: "Alpha", b_idx: "Beta", d_idx: "Delta"}
    ordered = sorted(
        records,
        key=lambda r: (
            0 if r["index"] == a_idx else (1 if r["index"] == b_idx else (2 if r["index"] == d_idx else 3)),
            -(r["t1"] + r["t2"] + r["t3"]),
            fitness_vals[r["index"]] if mode == "min" else -fitness_vals[r["index"]],
        ),
    )
    initial_audit = []
    for rank_i, rec in enumerate(ordered, start=1):
        wi = rec["index"]
        initial_audit.append({
            "rank": rank_i,
            "wolf": f"W-{wi + 1:02d}",
            "role": roles.get(wi, "Omega"),
            "t1": rec["t1"],
            "t2": rec["t2"],
            "t3": rec["t3"],
            "total_cols": rec["total_cols"],
            "fitness": round(float(fitness_vals[wi]), 5),
        })

    convergence_curve = []
    a_curve = []
    diversity_curve = []
    trajectories = []
    leaders = []

    for t in range(max_iter):
        if t > 0:
            for i in range(n_agents):
                positions[i] = np.clip(positions[i], lb, ub)
                fitness = float(obj_func(positions[i]))
                if is_better(fitness, best_fit, mode):
                    best_fit = fitness
                    best_pos = positions[i].copy()
                if is_better(fitness, alpha_score, mode):
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = fitness, positions[i].copy()
                elif is_better(fitness, beta_score, mode):
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = fitness, positions[i].copy()
                elif is_better(fitness, delta_score, mode):
                    delta_score, delta_pos = fitness, positions[i].copy()

        progress = t / max(1, max_iter - 1)
        a = 2.0 * (1.0 - progress)

        convergence_curve.append(float(best_fit))
        a_curve.append(float(a))
        diversity_curve.append(pack_diversity(positions, lb, ub))
        if record_2d:
            trajectories.append(positions[:, :2].round(4).tolist())
            leaders.append([
                alpha_pos[:2].round(4).tolist(),
                beta_pos[:2].round(4).tolist(),
                delta_pos[:2].round(4).tolist(),
            ])

        for i in range(n_agents):
            r1, r2 = rng.random(dim), rng.random(dim)
            A1, C1 = 2.0 * a * r1 - a, 2.0 * r2
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions[i])

            r1, r2 = rng.random(dim), rng.random(dim)
            A2, C2 = 2.0 * a * r1 - a, 2.0 * r2
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions[i])

            r1, r2 = rng.random(dim), rng.random(dim)
            A3, C3 = 2.0 * a * r1 - a, 2.0 * r2
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions[i])

            positions[i] = np.clip((X1 + X2 + X3) / 3.0, lb, ub)

    return RunResult(
        algo_id="dual_criteria_gwo",
        name="Dual-Criteria GWO",
        mode=mode,
        best_position=best_pos,
        best_fitness=float(best_fit),
        convergence_curve=convergence_curve,
        a_curve=a_curve,
        diversity_curve=diversity_curve,
        trajectory_history=trajectories,
        leaders_history=leaders,
        initial_pop_audit=initial_audit,
        runtime_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        convergence_iter=convergence_iteration(convergence_curve, mode),
    )