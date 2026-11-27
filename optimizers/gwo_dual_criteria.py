
import time
import numpy as np
from optimizers.common import (
    RunResult,
    is_better,
    pack_diversity,
    convergence_iteration,
)


def select_tiered_column_leaders(positions, fitness_vals, mode="min"):
    """
    Select Alpha, Beta and Delta using dual-criteria
    column-tier ranking.

    This method can be called at every iteration.
    """

    if mode not in ("min", "max"):
        raise ValueError("mode must be either 'min' or 'max'")

    n_wolves = len(positions)

    if n_wolves < 3:
        raise ValueError("Dual-Criteria GWO requires at least 3 wolves.")

    # Combine decision variables and fitness into one table.
    table = np.column_stack([positions, fitness_vals])

    n_wolves, n_cols = table.shape

    tier1 = np.zeros(n_wolves, dtype=int)
    tier2 = np.zeros(n_wolves, dtype=int)
    tier3 = np.zeros(n_wolves, dtype=int)

    # Calculate column-based tier scores.
    for c in range(n_cols):

        col = table[:, c]
        rounded_col = np.round(col, 12)

        unique_values = np.unique(rounded_col)

        ordered_values = (
            unique_values if mode == "min"
            else unique_values[::-1]
        )

        if len(ordered_values) >= 1:
            tier1 += (
                rounded_col == ordered_values[0]
            ).astype(int)

        if len(ordered_values) >= 2:
            tier2 += (
                rounded_col == ordered_values[1]
            ).astype(int)

        if len(ordered_values) >= 3:
            tier3 += (
                rounded_col == ordered_values[2]
            ).astype(int)

    remaining = list(range(n_wolves))

    # Fitness is used as a tie-breaker.
    def fit_key(idx):
        return (
            -fitness_vals[idx]
            if mode == "min"
            else fitness_vals[idx]
        )

    # Select Alpha.
    a_idx = max(
        remaining,
        key=lambda i: (
            int(tier1[i]),
            int(tier2[i]),
            int(tier3[i]),
            fit_key(i),
        ),
    )

    remaining.remove(a_idx)

    # Select Beta.
    b_idx = max(
        remaining,
        key=lambda i: (
            int(tier2[i]),
            int(tier1[i] + tier2[i]),
            int(tier3[i]),
            fit_key(i),
        ),
    )

    remaining.remove(b_idx)

    # Select Delta.
    d_idx = max(
        remaining,
        key=lambda i: (
            int(tier3[i]),
            int(tier1[i] + tier2[i] + tier3[i]),
            fit_key(i),
        ),
    )

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

    if mode not in ("min", "max"):
        raise ValueError("mode must be either 'min' or 'max'")

    if n_agents < 3:
        raise ValueError("n_agents must be at least 3")

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    rng = np.random.default_rng(seed)

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    dim = len(lb)

    if lb.shape != ub.shape or np.any(lb >= ub):
        raise ValueError("Invalid lower and upper bounds")

    # -------------------------------------------------
    # 1. INITIALIZE POPULATION
    # -------------------------------------------------

    positions = (
        lb + rng.random((n_agents, dim)) * (ub - lb)
    )

    fitness_vals = np.array(
        [float(obj_func(p)) for p in positions],
        dtype=float,
    )

    # Track the best solution separately.
    best_idx = (
        int(np.argmin(fitness_vals))
        if mode == "min"
        else int(np.argmax(fitness_vals))
    )

    best_pos = positions[best_idx].copy()
    best_fit = float(fitness_vals[best_idx])

    # -------------------------------------------------
    # 2. HISTORY VARIABLES
    # -------------------------------------------------

    convergence_curve = []
    a_curve = []
    diversity_curve = []

    trajectories = []
    leaders = []

    initial_audit = []

    # -------------------------------------------------
    # 3. MAIN OPTIMIZATION LOOP
    # -------------------------------------------------

    for t in range(max_iter):

        # Evaluate the population at the beginning
        # of every iteration.
        #
        # At t=0, fitness was already calculated during
        # initialization. At later iterations, wolves
        # have moved since the previous evaluation.

        if t > 0:

            fitness_vals = np.array(
                [float(obj_func(p)) for p in positions],
                dtype=float,
            )

            # Update global best-so-far solution.
            for i in range(n_agents):

                fitness = float(fitness_vals[i])

                if is_better(fitness, best_fit, mode):

                    best_fit = fitness
                    best_pos = positions[i].copy()

        # -------------------------------------------------
        # 4. APPLY DUAL-CRITERIA SELECTION EVERY ITERATION
        # -------------------------------------------------

        a_idx, b_idx, d_idx, records = (
            select_tiered_column_leaders(
                positions,
                fitness_vals,
                mode,
            )
        )

        # Assign the complete position vectors.
        alpha_pos = positions[a_idx].copy()
        beta_pos = positions[b_idx].copy()
        delta_pos = positions[d_idx].copy()

        # -------------------------------------------------
        # 5. INITIAL POPULATION AUDIT
        # -------------------------------------------------

        if t == 0:

            roles = {
                a_idx: "Alpha",
                b_idx: "Beta",
                d_idx: "Delta",
            }

            ordered = sorted(
                records,
                key=lambda r: (
                    0 if r["index"] == a_idx else (
                        1 if r["index"] == b_idx else (
                            2 if r["index"] == d_idx else 3
                        )
                    ),
                    -(
                        r["t1"] + r["t2"] + r["t3"]
                    ),
                    (
                        fitness_vals[r["index"]]
                        if mode == "min"
                        else -fitness_vals[r["index"]]
                    ),
                ),
            )

            for rank_i, rec in enumerate(
                ordered,
                start=1,
            ):

                wi = rec["index"]

                initial_audit.append({
                    "rank": rank_i,
                    "wolf": f"W-{wi + 1:02d}",
                    "role": roles.get(wi, "Omega"),
                    "t1": rec["t1"],
                    "t2": rec["t2"],
                    "t3": rec["t3"],
                    "total_cols": rec["total_cols"],
                    "fitness": round(
                        float(fitness_vals[wi]), 5
                    ),
                })

        # -------------------------------------------------
        # 6. CONVERGENCE FACTOR
        # -------------------------------------------------

        progress = t / max(1, max_iter - 1)

        # Standard linear GWO convergence factor.
        a = 2.0 * (1.0 - progress)

        # -------------------------------------------------
        # 7. RECORD OPTIMIZATION INFORMATION
        # -------------------------------------------------

        convergence_curve.append(float(best_fit))

        a_curve.append(float(a))

        diversity_curve.append(
            pack_diversity(positions, lb, ub)
        )

        if record_2d:

            trajectories.append(
                positions[:, :2].round(4).tolist()
            )

            leaders.append([
                alpha_pos[:2].round(4).tolist(),
                beta_pos[:2].round(4).tolist(),
                delta_pos[:2].round(4).tolist(),
            ])

        # -------------------------------------------------
        # 8. STANDARD GWO POSITION UPDATE
        # -------------------------------------------------

        new_positions = np.empty_like(positions)

        for i in range(n_agents):

            # Alpha leader
            r1, r2 = rng.random(dim), rng.random(dim)

            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2

            X1 = alpha_pos - A1 * np.abs(
                C1 * alpha_pos - positions[i]
            )

            # Beta leader
            r1, r2 = rng.random(dim), rng.random(dim)

            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2

            X2 = beta_pos - A2 * np.abs(
                C2 * beta_pos - positions[i]
            )

            # Delta leader
            r1, r2 = rng.random(dim), rng.random(dim)

            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2

            X3 = delta_pos - A3 * np.abs(
                C3 * delta_pos - positions[i]
            )

            # Average the three leader-guided positions.
            new_positions[i] = np.clip(
                (X1 + X2 + X3) / 3.0,
                lb,
                ub,
            )

        positions = new_positions

    # -------------------------------------------------
    # 9. FINAL RESULT
    # -------------------------------------------------

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
        runtime_ms=round(
            (time.perf_counter() - t0) * 1000.0,
            2,
        ),
        convergence_iter=convergence_iteration(
            convergence_curve, mode
        ),
    )