import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(log):
    t, pos, ref, err = log["t"], log["pos"], log["ref"], log["err"]

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(ref[:, 0], ref[:, 1], ref[:, 2], "--", label="reference (r)")
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="actual (y)")
    ax1.set_title("3D trajectory")
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(ref[:, 0], ref[:, 1], "--", label="reference (r)")
    ax2.plot(pos[:, 0], pos[:, 1], label="actual (y)")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Top-down (XY) view")
    ax2.axis("equal")
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, pos[:, 2], label="z actual")
    ax3.plot(t, ref[:, 2], "--", label="z reference")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("z [m]")
    ax3.set_title("Altitude tracking")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    err_norm = (err ** 2).sum(axis=1) ** 0.5
    ax4.plot(t, err_norm)
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("||e|| [m]")
    ax4.set_title("Position error norm")

    output_path = Path("simulation-images/simulation_result.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
