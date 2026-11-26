"""Publication plot of saved paired validation gaps; no new model evaluation."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parent / ".cache/paper-tooling/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SOURCE = HERE.parent / "results/eurosat_grouping_fp32_replication/cross_dataset_summary.json"


def main():
    study = json.loads(SOURCE.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35), layout="constrained")
    policies = [("role_100_0", "Roles preserved", "#26798e", "o"),
                ("free_100_0", "Roles free", "#bd641f", "^")]
    for ax, ds in zip(axes, study["datasets"]):
        for case, label, color, marker in policies:
            rows = [r for r in ds["paired_validation_gaps_pp"] if r["case"] == case]
            assert sorted(r["seed"] for r in rows) == [2026, 2027, 2028]
            values = np.array([[r["before"], r["after"]] for r in rows])
            for row in values:
                ax.plot([0, 1], row, color=color, marker=marker, alpha=.35,
                        linewidth=.7, markersize=3)
            ax.plot([0, 1], values.mean(0), color=color, marker=marker,
                    linewidth=2, markersize=5, label=label)
        ax.axhline(0, color=".45", linewidth=.6)
        ax.set_xticks([0, 1], ["Before repair", "After 3 epochs"])
        ax.set_xlim(-.12, 1.12)
        ax.set_ylim(bottom=0)
        ax.set_title(ds["name"], fontsize=10)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Original minus mixed accuracy (pp)")
    legend = [Line2D([0], [0], color=c, marker=m, label=l) for _, l, c, m in policies]
    axes[0].legend(handles=legend, loc="upper right", frameon=False, fontsize=8)
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    pdf = folder / "grouping_gaps.pdf"
    fig.savefig(pdf, metadata={"Creator": "paper/plot_overview.py", "CreationDate": None,
                              "ModDate": None, "Author": ""})
    fig.savefig(folder / "grouping_gaps.png")
    plt.close(fig)
    manifest = {"source": str(SOURCE.relative_to(HERE.parent)).replace("\\", "/"),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "descriptive_only": True, "new_inference": False}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Rendered paired grouping gaps from the existing cross-setting summary.")


if __name__ == "__main__":
    main()
