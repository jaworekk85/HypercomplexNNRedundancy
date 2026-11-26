"""Illustrative fixed-set assignments; this schematic contains no measured data."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parent / ".cache/paper-tooling/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    roles = "rijk"
    panels = [
        ("Original groups", [("rA", "iA", "jA", "kA"), ("rB", "iB", "jB", "kB")]),
        ("Role-preserving mix", [("rA", "iB", "jA", "kB"), ("rB", "iA", "jB", "kA")]),
        ("Role-free mix", [("iA", "rB", "kA", "jB"), ("kB", "jA", "rA", "iB")]),
    ]
    expected = sorted(role + source for source in "AB" for role in roles)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.38), layout="constrained")
    for ax, (title, groups) in zip(axes, panels):
        assert sorted(s for group in groups for s in group) == expected
        if title == "Role-preserving mix":
            assert all(s[0] == roles[i] for g in groups for i, s in enumerate(g))
        if title != "Original groups":
            assert all(len({s[1] for s in g}) == 2 for g in groups)
        ax.set_title(title, fontsize=10, pad=5)
        for column, role in enumerate(roles):
            ax.text(column + .5, 2.15, role, ha="center", va="center", fontsize=9)
        for row, group in enumerate(groups):
            y = 1 - row
            ax.text(-.1, y + .45, f"$g_{row+1}$", ha="right", va="center", fontsize=9)
            for column, component in enumerate(group):
                color = "#d5eaf0" if component[1] == "A" else "#f6dfc7"
                ax.add_patch(Rectangle((column, y), .95, .88, facecolor=color,
                                       edgecolor="#555555", linewidth=.6))
                ax.text(column + .475, y + .44, f"${component[0]}_{{{component[1]}}}$",
                        ha="center", va="center", fontsize=10)
        ax.set_xlim(-.45, 4.05)
        ax.set_ylim(-.15, 2.35)
        ax.axis("off")
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    pdf = folder / "grouping_concept.pdf"
    fig.savefig(pdf, metadata={"Creator": "paper/plot_grouping_concept.py", "Author": "",
                              "CreationDate": None, "ModDate": None})
    fig.savefig(folder / "grouping_concept.png", dpi=180)
    plt.close(fig)
    manifest = {"kind": "illustrative assignment diagram, not experimental data",
                "same_eight_distinct_components_each_panel": True,
                "assignments": [{"name": title, "groups": groups} for title, groups in panels],
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    (folder / "grouping_concept.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Rendered grouping schematic; retained component multisets verified.")


if __name__ == "__main__":
    main()
