"""Plot all saved validation epochs of the frozen matched repair study."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/paper-tooling/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = ROOT/'results/grouping_repair_curve'
    source = folder/'summary.json'
    study = json.loads(source.read_text())
    assert study['status'] == 'completed' and len(study['results']) == 18
    styles = [('original', 'Original groups', '#293f65'),
              ('role_100_0', 'Roles preserved', '#26798e'),
              ('free_100_0', 'Roles free', '#bd641f')]
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True, layout='constrained')
    exported = {}
    for column, (dataset, title) in enumerate([('cifar10', 'CIFAR-10 / ResNet-20'),
                                               ('eurosat', 'EuroSAT / CNN6')]):
        rows = {(r['seed'], r['case']): r for r in study['results'] if r['dataset'] == dataset}
        curves = {}
        for case, label, color in styles:
            values = []
            for seed in (2026, 2027, 2028):
                record = rows[(seed, case)]
                assert [r['epoch'] for r in record['history']] == list(range(1, 31))
                values.append([100*record['snapshots'][0]['metrics']['validation']['accuracy']] +
                              [100*r['validation']['accuracy'] for r in record['history']])
            curves[case] = np.array(values)
            for row in curves[case]:
                axes[0, column].plot(range(31), row, color=color, alpha=.19, linewidth=.7)
            axes[0, column].plot(range(31), curves[case].mean(0), color=color, linewidth=2,
                                 marker='o', markevery=[0, 1, 3, 10, 30], markersize=3, label=label)
        for case, label, color in styles[1:]:
            differences = curves['original'] - curves[case]
            for row in differences:
                axes[1, column].plot(range(31), row, color=color, alpha=.2, linewidth=.7)
            axes[1, column].plot(range(31), differences.mean(0), color=color, linewidth=2,
                                 marker='o', markevery=[0, 1, 3, 10, 30], markersize=3)
        axes[0, column].set_title(title)
        axes[1, column].axhline(0, color='.4', linewidth=.7, linestyle='--')
        axes[1, column].set_xlabel('Repair epochs (one common 30-epoch schedule)')
        axes[1, column].set_xticks([0, 3, 10, 20, 30])
        for ax in axes[:, column]:
            ax.grid(alpha=.16)
            ax.set_xlim(-.3, 30.3)
        exported[dataset] = {case: curves[case].tolist() for case, _, _ in styles}
    axes[0, 0].set_ylabel('Validation accuracy (%)')
    axes[1, 0].set_ylabel('Original minus mixed accuracy (pp)')
    axes[0, 0].legend(frameon=False, loc='lower right', fontsize=8)
    figures = folder/'figures'
    figures.mkdir(exist_ok=True)
    png, pdf = figures/'repair_curve.png', figures/'repair_curve.pdf'
    fig.savefig(png, dpi=180)
    fig.savefig(pdf, metadata={'Author': '', 'Creator': 'scripts/plot_grouping_repair_curve.py',
                              'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = {'source': source.relative_to(ROOT).as_posix(), 'source_sha256': sha(source),
              'figure_hashes': {p.name: sha(p) for p in (png, pdf)},
              'source_seeds': [2026, 2027, 2028], 'epochs': list(range(31)),
              'thin_lines': 'paired source seeds', 'thick_lines': 'descriptive arithmetic means',
              'no_significance_or_equivalence_claim': True, 'official_test_evaluated': False,
              'historical_three_epoch_schedule_not_plotted': True, 'accuracy_curves_percent': exported}
    (figures/'repair_curve.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(png.relative_to(ROOT).as_posix())


if __name__ == '__main__':
    main()
