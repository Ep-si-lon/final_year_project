"""Q14 deliverable: Bode plots (open-loop, with gain/phase margins marked)
and step-response plots (closed-loop) for each of the 5 linearized loops in
src/stability_analysis.py, plus a printed numeric margins/step-info table.

See src/stability_analysis.py's module docstring for why this is 5 separate
per-loop analyses rather than one Bode plot of "the whole system" -- the
full nonlinear trajectory-tracking loop isn't a single LTI system.

Not a pytest test -- run directly:
    python3 uav_trajectory_tracking/validation_tests/stability_analysis_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import control
import numpy as np

from uav_trajectory_tracking.src.controller import gains_pole_placement_outer
from uav_trajectory_tracking.src.params import CrazyflieParams
from uav_trajectory_tracking.src.stability_analysis import all_loops, margins_and_step_info

PLOTS_DIR = Path(__file__).parent / "plots"


def plot_bode_and_step(name: str, L, T, info: dict):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    w = np.logspace(-2, 3, 2000)
    resp = control.frequency_response(L, w)
    mag_db = 20 * np.log10(resp.magnitude)
    phase_deg = np.degrees(np.unwrap(resp.phase))

    axes[0].semilogx(w, mag_db)
    axes[0].axhline(0, color="gray", linewidth=0.7)
    axes[0].set_ylabel("magnitude [dB]")
    axes[0].set_xlabel("freq [rad/s]")
    axes[0].set_title(f"{name}: open-loop Bode (magnitude)")
    if np.isfinite(info["gain_crossover_freq_rad_s"]):
        axes[0].axvline(info["gain_crossover_freq_rad_s"], color="orange", linestyle="--", linewidth=1,
                         label=f"wc={info['gain_crossover_freq_rad_s']:.3g} rad/s")
        axes[0].legend(fontsize=8)

    axes[1].semilogx(w, phase_deg)
    # -180deg instability threshold: mark every angle congruent to 180 (mod
    # 360) within the plotted range, since np.unwrap's branch choice can put
    # the actual crossing at +180, -180, 540, etc. depending on the curve's
    # own starting phase -- a single hardcoded "-180" line can silently miss
    # the crossing entirely if unwrap picked a different branch (caught by
    # inspection: this loop's unwrapped phase runs ~90deg to ~270deg, so a
    # literal -180 line never intersects it even though it does cross the
    # equivalent +180 point).
    y_lo, y_hi = phase_deg.min() - 10, phase_deg.max() + 10
    n_lo, n_hi = int(np.floor((y_lo - 180) / 360)), int(np.ceil((y_hi - 180) / 360))
    for n in range(n_lo, n_hi + 1):
        axes[1].axhline(180 + 360 * n, color="gray", linewidth=0.7)
    axes[1].set_ylabel("phase [deg]")
    axes[1].set_xlabel("freq [rad/s]")
    axes[1].set_title(f"{name}: open-loop Bode (phase)\nGM={info['gain_margin_db']:.1f}dB  "
                       f"PM={info['phase_margin_deg']:.1f}deg")

    t_max = max(info["settling_time_s"] * 2.5, 1.0) if np.isfinite(info["settling_time_s"]) else 10.0
    t_step = np.linspace(0, t_max, 1000)
    resp_step = control.step_response(T, t_step)
    axes[2].plot(resp_step.time, resp_step.outputs)
    axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("closed-loop step response")
    axes[2].set_title(f"{name}: step response\nOvershoot={info['overshoot_pct']:.1f}%  "
                       f"Ts={info['settling_time_s']:.2f}s")

    fig.tight_layout()
    safe_name = name.replace("/", "-").replace(" ", "_").replace("(", "").replace(")", "")
    out = PLOTS_DIR / f"stability_{safe_name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    params = CrazyflieParams()
    gains = gains_pole_placement_outer(params)
    loops = all_loops(params, gains)

    header = f"{'loop':<22}{'GM [dB]':>10}{'PM [deg]':>10}{'wc [rad/s]':>12}{'Ts [s]':>10}{'OS [%]':>10}"
    print(header)
    print("-" * len(header))
    for name, (L, T) in loops.items():
        info = margins_and_step_info(L, T)
        print(f"{name:<22}{info['gain_margin_db']:>10.2f}{info['phase_margin_deg']:>10.2f}"
              f"{info['gain_crossover_freq_rad_s']:>12.4g}{info['settling_time_s']:>10.3f}"
              f"{info['overshoot_pct']:>10.2f}")
        plot_bode_and_step(name, L, T, info)

    print("\nNote: 'x/y (outer position)' is a type-3 (conditionally stable) loop -- its raw\n"
          "gain margin reads negative even though it IS stable at the design gains (verified\n"
          "via closed-loop pole locations in tests/test_stability_analysis.py). See\n"
          "src/stability_analysis.py's outer_position_loop() docstring.")


if __name__ == "__main__":
    main()
