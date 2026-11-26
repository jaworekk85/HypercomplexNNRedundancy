"""Descriptive validation-accuracy figure; source seeds remain visible."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'results/grouping_bn_accuracy_control'
    summary = json.loads((output/'summary.json').read_text())
    if summary['status'] != 'completed' or summary['completed_models'] != 18:
        raise ValueError('Figure requires all 18 models')
    cases = ['parent', 'original', 'role_50_0', 'free_50_0', 'role_100_0', 'free_100_0']
    labels = ['Parent', 'Original', 'Role 50%', 'Free 50%', 'Role 100%', 'Free 100%']
    fig, ax = plt.subplots(figsize=(10, 5))
    for stage, offset, color, title in [('native', -.23, '#7e8a97', 'Before repair'),
                                        ('bn_only', 0, '#ce8629', 'BN only'),
                                        ('historical_repair', .23, '#287d8e', 'Historical 3-epoch repair')]:
        for position, case in enumerate(cases):
            values = [100*r[stage]['validation']['accuracy'] for r in summary['results']
                      if r['case'] == case and r[stage] is not None]
            if not values:
                continue
            x = position + offset
            ax.bar(x, np.mean(values), width=.21, color=color, alpha=.85,
                   label=title if case == 'original' else None)
            ax.scatter(x+np.linspace(-.04, .04, len(values)), values, color=color,
                       edgecolors='black', linewidths=.5, s=18, zorder=3)
    ax.set_xticks(range(6), labels)
    ax.set_ylabel('Validation accuracy (%)')
    ax.set_ylim(65, 95)
    ax.set_title('BN-only intervention on existing quaternion models')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', frameon=False)
    fig.text(.5, .015, 'Bars: means of 3 source seeds. Dots: individual seeds. Previously explored validation split.\n'
             'BN only: unchanged trainable weights. Historical repair did not start from the new BN state.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(folder/f'validation_accuracy.{extension}', dpi=180, bbox_inches='tight')
    metadata = {'matplotlib': matplotlib.__version__, 'source_summary_sha256': file_sha256(output/'summary.json'),
                'script_sha256': file_sha256(Path(__file__)), 'descriptive_only': True}
    (folder/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(folder.relative_to(ROOT))


if __name__ == '__main__':
    main()
