"""Tune the shared outer-loop X/Y PID gains with Particle Swarm Optimization.

The nonlinear plant, motor lag, circular reference, altitude controller and
inner attitude controller are held fixed.  Only [Kp, Ki, Kd] in the outer
position loop are optimized.  The objective rewards low circle-tracking error
after the take-off transient and penalizes sustained actuator saturation.

Run from this directory:
    python pso_tune.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynamics import CrazyflieParams
from simulate import run


GAIN_NAMES = ("kp", "ki", "kd")
DEFAULT_BOUNDS = np.array([
    [0.05, 3.00],  # Kp
    [0.00, 0.80],  # Ki
    [0.05, 2.50],  # Kd
])
TRAINING_DISTURBANCE = np.array([0.12, -0.08, 0.0])  # m/s^2, steady crosswind/model bias


@dataclass(frozen=True)
class TrackingMetrics:
    xy_rmse_m: float
    steady_xy_rmse_m: float
    steady_xy_mean_m: float
    z_rmse_m: float
    max_tilt_deg: float
    torque_saturation_fraction: float


def gain_dict(values):
    return dict(zip(GAIN_NAMES, map(float, values)))


def tracking_metrics(log, steady_start=8.0):
    error = log["ref"] - log["pos"]
    xy_error = np.linalg.norm(error[:, :2], axis=1)
    steady = log["t"] >= steady_start
    if not np.any(steady):
        steady = np.ones_like(log["t"], dtype=bool)

    params = CrazyflieParams()
    torques = np.abs(log["u_actual"][:, 1:])
    saturated = np.any(torques >= 0.995 * params.max_torque, axis=1)

    return TrackingMetrics(
        xy_rmse_m=float(np.sqrt(np.mean(xy_error ** 2))),
        steady_xy_rmse_m=float(np.sqrt(np.mean(xy_error[steady] ** 2))),
        steady_xy_mean_m=float(np.mean(xy_error[steady])),
        z_rmse_m=float(np.sqrt(np.mean(error[:, 2] ** 2))),
        max_tilt_deg=float(np.rad2deg(np.max(np.abs(log["att"][:, :2])))),
        torque_saturation_fraction=float(np.mean(saturated)),
    )


def objective_from_metrics(metrics):
    """Scalar cost: tracking dominates, saturation discourages unsafe gains."""
    return (
        0.35 * metrics.xy_rmse_m
        + 1.50 * metrics.steady_xy_rmse_m
        + 1.50 * metrics.steady_xy_mean_m
        + 0.05 * metrics.z_rmse_m
        + 0.50 * metrics.torque_saturation_fraction
        + 0.02 * max(0.0, metrics.max_tilt_deg - 20.0) ** 2
    )


def evaluate(values, duration=20.0, dt=0.01, steady_start=8.0):
    try:
        log = run(
            duration=duration,
            dt=dt,
            trajectory="circle",
            gain_preset="margin_safe",
            outer_gains=gain_dict(values),
            disturbance_accel=TRAINING_DISTURBANCE,
        )
        arrays = (log["pos"], log["att"], log["u_actual"])
        if not all(np.all(np.isfinite(a)) for a in arrays):
            return 1e6, None
        if np.max(np.abs(log["pos"])) > 20.0 or np.max(np.abs(log["att"])) > np.pi:
            return 1e6, None
        metrics = tracking_metrics(log, steady_start=steady_start)
        return objective_from_metrics(metrics), metrics
    except (FloatingPointError, OverflowError, ValueError):
        return 1e6, None


def particle_swarm(objective, bounds=DEFAULT_BOUNDS, particles=18, iterations=24,
                   seed=7, initial_positions=None):
    """Minimize a bounded objective with canonical global-best PSO."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (3, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds must have shape (3, 2) with lower < upper")
    if particles < 2 or iterations < 1:
        raise ValueError("particles must be >= 2 and iterations must be >= 1")

    rng = np.random.default_rng(seed)
    lower, upper = bounds[:, 0], bounds[:, 1]
    span = upper - lower
    position = rng.uniform(lower, upper, size=(particles, 3))
    velocity = rng.uniform(-0.15 * span, 0.15 * span, size=(particles, 3))

    if initial_positions is not None:
        seeds = np.atleast_2d(np.asarray(initial_positions, dtype=float))
        count = min(len(seeds), particles)
        position[:count] = np.clip(seeds[:count], lower, upper)
        velocity[:count] = 0.0

    personal_position = position.copy()
    personal_cost = np.full(particles, np.inf)
    global_position = position[0].copy()
    global_cost = np.inf
    history = []

    for iteration in range(iterations):
        for index in range(particles):
            cost = float(objective(position[index]))
            if cost < personal_cost[index]:
                personal_cost[index] = cost
                personal_position[index] = position[index].copy()
            if cost < global_cost:
                global_cost = cost
                global_position = position[index].copy()

        history.append(global_cost)
        print(
            f"iteration {iteration + 1:02d}/{iterations}: cost={global_cost:.6f}, "
            f"Kp={global_position[0]:.6f}, Ki={global_position[1]:.6f}, "
            f"Kd={global_position[2]:.6f}",
            flush=True,
        )

        inertia = 0.90 - 0.50 * iteration / max(1, iterations - 1)
        r1 = rng.random((particles, 3))
        r2 = rng.random((particles, 3))
        velocity = (
            inertia * velocity
            + 1.60 * r1 * (personal_position - position)
            + 1.60 * r2 * (global_position - position)
        )
        velocity = np.clip(velocity, -0.25 * span, 0.25 * span)
        position = np.clip(position + velocity, lower, upper)

    return global_position, global_cost, np.asarray(history)


def comparison_plot(baseline_log, optimized_log, history, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(baseline_log["ref"][:, 0], baseline_log["ref"][:, 1], "k--", label="reference")
    ax.plot(baseline_log["pos"][:, 0], baseline_log["pos"][:, 1], label="formula PID")
    ax.plot(optimized_log["pos"][:, 0], optimized_log["pos"][:, 1], label="PSO PID")
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Circular trajectory with constant bias")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    for name, log in (("formula PID", baseline_log), ("PSO PID", optimized_log)):
        xy_error = np.linalg.norm(log["ref"][:, :2] - log["pos"][:, :2], axis=1)
        ax.plot(log["t"], xy_error, label=name)
    ax.axvline(8.0, color="k", linestyle=":", linewidth=1, label="steady-state scoring starts")
    ax.set(xlabel="time [s]", ylabel="XY error [m]", title="Position tracking error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(baseline_log["t"], baseline_log["ref"][:, 2], "k--", label="reference")
    ax.plot(baseline_log["t"], baseline_log["pos"][:, 2], label="formula PID")
    ax.plot(optimized_log["t"], optimized_log["pos"][:, 2], label="PSO PID")
    ax.set(xlabel="time [s]", ylabel="z [m]", title="Altitude (unchanged controller)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(np.arange(1, len(history) + 1), history, marker="o", markersize=3)
    ax.set(xlabel="PSO iteration", ylabel="best objective", title="PSO convergence")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=18)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=20.0,
                        help="training simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--validation-duration", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("pso_results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formula_gains = np.array([0.5 ** 2, 0.5 ** 3, 2 * 0.9 * 0.5])
    hand_tuned_gains = np.array([0.3, 0.02, 0.8])

    def cost(values):
        value, _ = evaluate(values, duration=args.duration, dt=args.dt)
        return value

    best, best_cost, history = particle_swarm(
        cost,
        particles=args.particles,
        iterations=args.iterations,
        seed=args.seed,
        initial_positions=[formula_gains, hand_tuned_gains],
    )
    optimized_gains = gain_dict(best)

    baseline_log = run(
        duration=args.validation_duration,
        dt=args.dt,
        gain_preset="margin_safe",
    )
    optimized_log = run(
        duration=args.validation_duration,
        dt=args.dt,
        gain_preset="margin_safe",
        outer_gains=optimized_gains,
    )
    disturbed_baseline_log = run(
        duration=args.validation_duration,
        dt=args.dt,
        gain_preset="margin_safe",
        disturbance_accel=TRAINING_DISTURBANCE,
    )
    disturbed_optimized_log = run(
        duration=args.validation_duration,
        dt=args.dt,
        gain_preset="margin_safe",
        outer_gains=optimized_gains,
        disturbance_accel=TRAINING_DISTURBANCE,
    )
    baseline_metrics = tracking_metrics(baseline_log)
    optimized_metrics = tracking_metrics(optimized_log)
    disturbed_baseline_metrics = tracking_metrics(disturbed_baseline_log)
    disturbed_optimized_metrics = tracking_metrics(disturbed_optimized_log)

    result = {
        "seed": args.seed,
        "particles": args.particles,
        "iterations": args.iterations,
        "training_duration_s": args.duration,
        "validation_duration_s": args.validation_duration,
        "dt_s": args.dt,
        "bounds": dict(zip(GAIN_NAMES, DEFAULT_BOUNDS.tolist())),
        "best_objective": best_cost,
        "optimized_gains": optimized_gains,
        "formula_gains": gain_dict(formula_gains),
        "formula_metrics": asdict(baseline_metrics),
        "optimized_metrics": asdict(optimized_metrics),
        "training_disturbance_accel_mps2": TRAINING_DISTURBANCE.tolist(),
        "disturbed_formula_metrics": asdict(disturbed_baseline_metrics),
        "disturbed_optimized_metrics": asdict(disturbed_optimized_metrics),
        "convergence": history.tolist(),
    }
    json_path = args.output_dir / "pso_results.json"
    plot_path = args.output_dir / "pso_comparison.png"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    comparison_plot(disturbed_baseline_log, disturbed_optimized_log, history, plot_path)

    print("\nOptimized gains:")
    print(json.dumps(optimized_gains, indent=2))
    print("\n60 s validation metrics:")
    print("formula PID:", json.dumps(asdict(baseline_metrics), indent=2))
    print("PSO PID:    ", json.dumps(asdict(optimized_metrics), indent=2))
    print("\n60 s validation metrics with constant horizontal disturbance:")
    print("formula PID:", json.dumps(asdict(disturbed_baseline_metrics), indent=2))
    print("PSO PID:    ", json.dumps(asdict(disturbed_optimized_metrics), indent=2))
    print(f"\nSaved {json_path} and {plot_path}")


if __name__ == "__main__":
    main()
