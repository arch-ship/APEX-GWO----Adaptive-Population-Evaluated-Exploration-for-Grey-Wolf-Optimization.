
from dataclasses import dataclass
from typing import Any, Callable, Dict
import numpy as np


@dataclass
class ProblemSpec:
    id: str
    name: str
    category: str
    mode: str
    dim: int
    lb: np.ndarray
    ub: np.ndarray
    obj_func: Callable[[np.ndarray], float]
    pop_value_func: Callable[[np.ndarray], float]
    description: str
    unit: str
    decode_func: Callable[[np.ndarray], Dict[str, Any]]
    optimum_2d: float = 0.0


# ---------------------------------------------------------
# Shared constrained-problem helpers
# ---------------------------------------------------------

PENALTY_FACTOR = 1.0e6


def _constraint_penalty(objective: float, constraints) -> float:
    """
    All constraints must be expressed as g(x) <= 0.
    Positive values represent constraint violations.

    The same penalty rule is used for every engineering
    problem to keep comparisons consistent.
    """
    g = np.asarray(constraints, dtype=float)
    violation = np.maximum(g, 0.0)

    penalty = PENALTY_FACTOR * float(np.sum(violation ** 2))
    return float(objective + penalty)


def _make_engineering_problem(
    pid,
    name,
    dim,
    lb,
    ub,
    objective,
    constraints,
    description,
    unit,
    decoder,
    category="Engineering Design",
    evaluator=None,
):
    """
    Creates a ProblemSpec while preserving the existing API.

    evaluator(x), when supplied, returns the design variables
    actually used by the engineering equations. It is useful
    for integer or repaired variables.
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    def prepare(x):
        x = np.clip(np.asarray(x, dtype=float), lb, ub)
        if evaluator is not None:
            x = np.asarray(evaluator(x), dtype=float)
        return x

    def raw_objective(x):
        x = prepare(x)
        return float(objective(x))

    def obj_func(x):
        x = prepare(x)
        f = float(objective(x))
        g = constraints(x)
        return _constraint_penalty(f, g)

    def pop_value_func(x):
        # The population metric shows the actual engineering
        # objective, not the penalty-augmented fitness.
        return raw_objective(x)

    def decode_func(x):
        x = prepare(x)
        result = decoder(x)
        result["objective"] = round(float(objective(x)), 8)
        result["penalized_fitness"] = round(float(obj_func(x)), 8)
        result["constraint_violation"] = round(
            float(np.sum(np.maximum(np.asarray(constraints(x)), 0.0))),
            8,
        )
        result["feasible"] = bool(
            np.all(np.asarray(constraints(x), dtype=float) <= 1e-8)
        )
        return result

    return ProblemSpec(
        id=pid,
        name=name,
        category=category,
        mode="min",
        dim=dim,
        lb=lb,
        ub=ub,
        obj_func=obj_func,
        pop_value_func=pop_value_func,
        description=description,
        unit=unit,
        decode_func=decode_func,
    )


# ---------------------------------------------------------
# 1. Tension / Compression Spring Design
# ---------------------------------------------------------

def _make_spring():
    """
    Variables:
      x1 = wire diameter
      x2 = mean coil diameter
      x3 = number of active coils

    Objective: minimize spring weight.
    Four standard inequality constraints.
    """

    def objective(x):
        d, D, N = x
        return (N + 2.0) * D * d ** 2

    def constraints(x):
        d, D, N = x

        g1 = 1.0 - (D ** 3 * N) / (71785.0 * d ** 4)

        denominator = 12566.0 * (D * d ** 3 - d ** 4)
        denominator = max(denominator, 1e-12)

        g2 = (
            (4.0 * D ** 2 - d * D) / denominator
            + 1.0 / (5108.0 * d ** 2)
            - 1.0
        )

        g3 = 1.0 - (140.45 * d) / (D ** 2 * N)
        g4 = (d + D) / 1.5 - 1.0

        return [g1, g2, g3, g4]

    def decode(x):
        d, D, N = x
        return {
            "type": "engineering",
            "design": "Tension/Compression Spring",
            "Wire Diameter": round(float(d), 5),
            "Mean Coil Diameter": round(float(D), 5),
            "Active Coils": round(float(N), 5),
            "Spring Weight Index": round(float(objective(x)), 8),
        }

    return _make_engineering_problem(
        pid="spring_design",
        name="Tension/Compression Spring Design",
        dim=3,
        lb=[0.05, 0.25, 2.0],
        ub=[2.0, 1.3, 15.0],
        objective=objective,
        constraints=constraints,
        description=(
            "Minimize spring weight while satisfying shear-stress, "
            "deflection, surge-frequency and geometric constraints."
        ),
        unit="Weight Index",
        decoder=decode,
    )


# ---------------------------------------------------------
# 2. Pressure Vessel Design
# ---------------------------------------------------------

def _make_pressure_vessel():
    """
    Variables:
      x1 = shell thickness
      x2 = head thickness
      x3 = inner radius
      x4 = cylindrical section length

    This version uses continuous thickness variables.
    The traditional benchmark uses 0.0625 thickness increments.
    """

    def objective(x):
        Ts, Th, R, L = x

        return (
            0.6224 * Ts * R * L
            + 1.7781 * Th * R ** 2
            + 3.1661 * Ts ** 2 * L
            + 19.84 * Ts ** 2 * R
        )

    def constraints(x):
        Ts, Th, R, L = x

        g1 = -Ts + 0.0193 * R
        g2 = -Th + 0.00954 * R
        g3 = (
            -(np.pi * R ** 2 * L)
            - (4.0 / 3.0) * np.pi * R ** 3
            + 1296000.0
        )
        g4 = L - 240.0

        # Normalize constraints to reduce scale imbalance.
        return [
            g1 / 10.0,
            g2 / 10.0,
            g3 / 1.0e6,
            g4 / 200.0,
        ]

    def decode(x):
        Ts, Th, R, L = x
        return {
            "type": "engineering",
            "design": "Pressure Vessel",
            "Shell Thickness": round(float(Ts), 5),
            "Head Thickness": round(float(Th), 5),
            "Inner Radius": round(float(R), 5),
            "Cylinder Length": round(float(L), 5),
            "Manufacturing Cost": round(float(objective(x)), 5),
        }

    return _make_engineering_problem(
        pid="pressure_vessel",
        name="Pressure Vessel Design",
        dim=4,
        lb=[0.0625, 0.0625, 10.0, 10.0],
        ub=[99.0, 99.0, 200.0, 200.0],
        objective=objective,
        constraints=constraints,
        description=(
            "Minimize pressure-vessel manufacturing cost while "
            "satisfying shell thickness, head thickness, volume "
            "and vessel-length constraints."
        ),
        unit="Cost",
        decoder=decode,
    )


# ---------------------------------------------------------
# 3. Three-Bar Truss Design
# ---------------------------------------------------------

def _make_three_bar_truss():
    """
    Variables:
      x1 = cross-sectional area of bars 1 and 3
      x2 = cross-sectional area of bar 2

    Objective: minimize structural weight.
    """

    P = 2.0
    sigma = 2.0
    length = 100.0

    def objective(x):
        A1, A2 = x
        return length * (2.0 * np.sqrt(2.0) * A1 + A2)

    def constraints(x):
        A1, A2 = x

        denominator = np.sqrt(2.0) * A1 ** 2 + 2.0 * A1 * A2
        denominator = max(denominator, 1e-12)

        g1 = (
            (np.sqrt(2.0) * A1 + A2) / denominator
        ) * P - sigma

        g2 = (A2 / denominator) * P - sigma

        denominator3 = np.sqrt(2.0) * A2 + A1
        denominator3 = max(denominator3, 1e-12)

        g3 = (1.0 / denominator3) * P - sigma

        return [g1 / 2.0, g2 / 2.0, g3 / 2.0]

    def decode(x):
        A1, A2 = x
        return {
            "type": "engineering",
            "design": "Three-Bar Truss",
            "Area A1": round(float(A1), 6),
            "Area A2": round(float(A2), 6),
            "Structural Weight": round(float(objective(x)), 6),
        }

    return _make_engineering_problem(
        pid="three_bar_truss",
        name="Three-Bar Truss Design",
        dim=2,
        lb=[0.01, 0.01],
        ub=[1.0, 1.0],
        objective=objective,
        constraints=constraints,
        description=(
            "Minimize the weight of a three-bar truss while "
            "satisfying its structural stress constraints."
        ),
        unit="Weight",
        decoder=decode,
    )


# ---------------------------------------------------------
# 4. Welded Beam Design
# ---------------------------------------------------------

def _make_welded_beam():
    """
    Variables:
      x1 = weld thickness
      x2 = weld length
      x3 = beam height
      x4 = beam thickness
    """

    P = 6000.0
    L = 14.0
    E = 30.0e6
    G = 12.0e6

    tau_max = 13600.0
    sigma_max = 30000.0
    delta_max = 0.25

    def engineering_values(x):
        h, ell, t, b = x

        tau1 = P / (np.sqrt(2.0) * h * ell)
        M = P * (L + ell / 2.0)

        R = np.sqrt(ell ** 2 / 4.0 + ((h + t) / 2.0) ** 2)

        J = (
            2.0 * np.sqrt(2.0) * h * ell
            * (ell ** 2 / 12.0 + ((h + t) / 2.0) ** 2)
        )
        J = max(J, 1e-12)

        tau2 = M * R / J

        tau = np.sqrt(
            tau1 ** 2
            + 2.0 * tau1 * tau2 * ell / (2.0 * R + 1e-12)
            + tau2 ** 2
        )

        sigma = 6.0 * P * L / (b * t ** 2)

        delta = 4.0 * P * L ** 3 / (E * b * t ** 3)

        Pc = (
            4.013 * E
            * np.sqrt(t ** 2 * b ** 6 / 36.0)
            / L ** 2
            * (
                1.0
                - t / (2.0 * L)
                * np.sqrt(E / (4.0 * G))
            )
        )

        return tau, sigma, delta, Pc

    def objective(x):
        h, ell, t, b = x
        return 1.10471 * h ** 2 * ell + 0.04811 * t * b * (14.0 + ell)

    def constraints(x):
        h, ell, t, b = x
        tau, sigma, delta, Pc = engineering_values(x)

        g1 = tau / tau_max - 1.0
        g2 = sigma / sigma_max - 1.0
        g3 = h / b - 1.0
        g4 = (0.10471 * h ** 2 + 0.04811 * t * b * (14.0 + ell)) / 5.0 - 1.0
        g5 = 0.125 / h - 1.0
        g6 = delta / delta_max - 1.0
        g7 = 6000.0 / max(Pc, 1e-12) - 1.0

        return [g1, g2, g3, g4, g5, g6, g7]

    def decode(x):
        h, ell, t, b = x
        tau, sigma, delta, Pc = engineering_values(x)

        return {
            "type": "engineering",
            "design": "Welded Beam",
            "Weld Thickness": round(float(h), 6),
            "Weld Length": round(float(ell), 6),
            "Beam Height": round(float(t), 6),
            "Beam Thickness": round(float(b), 6),
            "Shear Stress": round(float(tau), 4),
            "Bending Stress": round(float(sigma), 4),
            "Deflection": round(float(delta), 7),
            "Buckling Load": round(float(Pc), 4),
        }

    return _make_engineering_problem(
        pid="welded_beam",
        name="Welded Beam Design",
        dim=4,
        lb=[0.1, 0.1, 0.1, 0.1],
        ub=[2.0, 10.0, 10.0, 2.0],
        objective=objective,
        constraints=constraints,
        description=(
            "Minimize welded-beam fabrication cost while satisfying "
            "shear stress, bending stress, geometry, deflection "
            "and buckling constraints."
        ),
        unit="Cost",
        decoder=decode,
    )


# ---------------------------------------------------------
# 5. Speed Reducer Design
# ---------------------------------------------------------

def _make_speed_reducer():
    """
    Seven-variable speed reducer benchmark.

    x3 (number of gear teeth) is rounded to an integer during
    evaluation. The remaining variables are continuous.
    """

    def repair(x):
        x = np.asarray(x, dtype=float).copy()
        x[2] = np.clip(np.round(x[2]), 17, 28)
        return x

    def objective(x):
        x1, x2, x3, x4, x5, x6, x7 = x

        return (
            0.7854 * x1 * x2 ** 2
            * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
            + 7.4777 * (x6 ** 3 + x7 ** 3)
            + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2)
        )

    def constraints(x):
        x1, x2, x3, x4, x5, x6, x7 = x

        g1 = 27.0 / (x1 * x2 ** 2 * x3) - 1.0
        g2 = 397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1.0
        g3 = 1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1.0
        g4 = 1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1.0

        g5 = (
            np.sqrt(
                (745.0 * x4 / (x2 * x3)) ** 2
                + 16.9e6
            ) / (110.0 * x6 ** 3)
            - 1.0
        )

        g6 = (
            np.sqrt(
                (745.0 * x5 / (x2 * x3)) ** 2
                + 157.5e6
            ) / (85.0 * x7 ** 3)
            - 1.0
        )

        g7 = x2 * x3 / 40.0 - 1.0
        g8 = 5.0 * x2 / x1 - 1.0
        g9 = x1 / (12.0 * x2) - 1.0
        g10 = (1.5 * x6 + 1.9) / x4 - 1.0
        g11 = (1.1 * x7 + 1.9) / x5 - 1.0

        return [
            g1, g2, g3, g4, g5, g6,
            g7, g8, g9, g10, g11,
        ]

    def decode(x):
        return {
            "type": "engineering",
            "design": "Speed Reducer",
            "Face Width": round(float(x[0]), 5),
            "Module": round(float(x[1]), 5),
            "Gear Teeth": int(round(float(x[2]))),
            "Bearing Length 1": round(float(x[3]), 5),
            "Bearing Length 2": round(float(x[4]), 5),
            "Shaft Diameter 1": round(float(x[5]), 5),
            "Shaft Diameter 2": round(float(x[6]), 5),
            "Reducer Weight": round(float(objective(x)), 5),
        }

    return _make_engineering_problem(
        pid="speed_reducer",
        name="Speed Reducer Design",
        dim=7,
        lb=[2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0],
        ub=[3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5],
        objective=objective,
        constraints=constraints,
        description=(
            "Minimize speed-reducer weight while satisfying gear "
            "geometry, shaft strength, bending stress and "
            "design constraints."
        ),
        unit="Weight",
        decoder=decode,
        evaluator=repair,
    )


# ---------------------------------------------------------
# 6. Cantilever Beam Design
# ---------------------------------------------------------

def _make_cantilever_beam():
    """
    Five-segment cantilever beam benchmark.

    Variables represent the widths of five beam segments.
    Objective: minimize total beam weight.
    """

    def objective(x):
        return 0.0624 * float(np.sum(x))

    def constraints(x):
        g = (
            61.0 / x[0] ** 3
            + 37.0 / x[1] ** 3
            + 19.0 / x[2] ** 3
            + 7.0 / x[3] ** 3
            + 1.0 / x[4] ** 3
            - 1.0
        )
        return [g]

    def decode(x):
        return {
            "type": "engineering",
            "design": "Cantilever Beam",
            "Segment Widths": [round(float(v), 6) for v in x],
            "Beam Weight": round(float(objective(x)), 8),
            "Deflection Constraint Value": round(
                float(constraints(x)[0] + 1.0), 8
            ),
        }

    return _make_engineering_problem(
        pid="cantilever_beam",
        name="Cantilever Beam Design",
        dim=5,
        lb=np.full(5, 0.01),
        ub=np.full(5, 100.0),
        objective=objective,
        constraints=constraints,
        description=(
            "Minimize cantilever-beam weight by selecting five "
            "segment widths while satisfying the deflection limit."
        ),
        unit="Weight",
        decoder=decode,
    )


# ---------------------------------------------------------
# Build the engineering problem suite
# ---------------------------------------------------------

def build_problem_suite(seed=2026) -> Dict[str, ProblemSpec]:
    """
    Returns the six engineering design benchmarks.

    The seed argument is retained for compatibility with the
    previous suite and existing application calls.
    """
    items = [
        _make_spring(),
        _make_pressure_vessel(),
        _make_three_bar_truss(),
        _make_welded_beam(),
        _make_speed_reducer(),
        _make_cantilever_beam(),
    ]

    return {p.id: p for p in items}


# ---------------------------------------------------------
# Existing 2D mathematical benchmarks
# Keep these for visualization only.
# ---------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    return float(
        10.0 * len(x)
        + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))
    )


def ackley(x: np.ndarray) -> float:
    n = len(x)
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n)
        + 20.0
        + np.e
    )


def rosenbrock(x: np.ndarray) -> float:
    return float(
        np.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2
            + (1.0 - x[:-1]) ** 2
        )
    )


def _make_continuous(pid, name, fn, bound, dim=2, opt_val=0.0):
    return ProblemSpec(
        id=pid,
        name=f"{name} ({dim}D)",
        category="Continuous",
        mode="min",
        dim=dim,
        lb=np.full(dim, -bound),
        ub=np.full(dim, bound),
        obj_func=fn,
        pop_value_func=lambda x: float(
            np.sqrt(np.mean((x - opt_val) ** 2))
        ),
        description=(
            f"Minimize {name} benchmark landscape in {dim} dimensions."
        ),
        unit="Loss",
        decode_func=lambda x: {
            "type": "kv",
            "Loss": round(float(fn(x)), 6),
            "L2 Norm": round(float(np.linalg.norm(x)), 4),
        },
        optimum_2d=opt_val,
    )


def build_2d_benchmarks() -> Dict[str, ProblemSpec]:
    items = [
        _make_continuous("rastrigin_2d", "Rastrigin", rastrigin, 5.12, 2),
        _make_continuous("ackley_2d", "Ackley", ackley, 8.0, 2),
        _make_continuous("rosenbrock_2d", "Rosenbrock", rosenbrock, 3.0, 2, 1.0),
        _make_continuous("sphere_2d", "Sphere", sphere, 6.0, 2),
    ]
    return {p.id: p for p in items}