"""Descriptive cross-dataset figure/tables from completed grouping studies."""
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CASES = ['original', 'role_100_0', 'free_100_0']


def main():
    specs = [('CIFAR-10 / ResNet-20', 'results/grouping_mechanism_study',
              'results/resnet20_baselines/quaternion/metrics.json'),
             ('EuroSAT / CNN6', 'results/eurosat_grouping_fp32_replication',
              'results/eurosat_cnn/seed_2026/quaternion/metrics.json')]
    datasets, summaries, paths = [], [], []
    for name, folder, parent_path in specs:
        path = ROOT/folder/'summary.json'
        result = json.loads(path.read_text())
        if result['status'] != 'completed':
            raise ValueError('Cross-dataset figure requires both completed studies')
        rows = [r for r in result['results'] if r['case'] in CASES]
        if len(rows) != 9 or {r['seed'] for r in rows} != {2026, 2027, 2028}:
            raise AssertionError('Missing paired source seeds/conditions')
        parent_count = json.loads((ROOT/parent_path).read_text())['real_trainable_parameters']
        compact_counts = {r['real_trainable_parameters'] for r in rows}
        if len(compact_counts) != 1:
            raise AssertionError('Unmatched compact parameter counts')
        compact = next(iter(compact_counts))
        aggregate = [{'case': case,
                      **{stage: {'accuracy_percent': 100*mean(r[stage]['validation']['accuracy'] for r in rows if r['case'] == case),
                                 'cross_entropy': mean(r[stage]['validation']['cross_entropy'] for r in rows if r['case'] == case)}
                         for stage in ('before', 'after')}} for case in CASES]
        gaps = []
        for seed in (2026, 2027, 2028):
            source = {r['case']: r for r in rows if r['seed'] == seed}
            for case in CASES[1:]:
                gaps.append({'seed': seed, 'case': case,
                             **{stage: 100*(source['original'][stage]['validation']['accuracy']-source[case][stage]['validation']['accuracy'])
                                for stage in ('before', 'after')}})
        entry = {'name': name, 'parent_real_parameters': parent_count, 'compact_real_parameters': compact,
                 'whole_model_savings_percent': 100*(1-compact/parent_count), 'source_seed_count': 3,
                 'aggregates': aggregate, 'paired_validation_gaps_pp': gaps,
                 'attempted_updates_per_model': sorted({r['optimizer_counts']['attempted'] for r in rows}),
                 'successful_update_range': [min(r['optimizer_counts']['successful'] for r in rows),
                                              max(r['optimizer_counts']['successful'] for r in rows)]}
        datasets.append(entry)
        summaries.append(rows)
        paths.append(path)
    output = ROOT/specs[1][1]
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharey='row')
    for column, (dataset, rows) in enumerate(zip(datasets, summaries)):
        for stage, offset, color, label in [('before', -.17, '#a5acb5', 'Native compact'),
                                          ('after', .17, '#277e89', 'After 3 repair epochs')]:
            for position, case in enumerate(CASES):
                values = [100*r[stage]['validation']['accuracy'] for r in rows if r['case'] == case]
                x = position+offset
                axes[0, column].bar(x, mean(values), .3, color=color, label=label if position == 0 else None)
                axes[0, column].scatter(x+np.linspace(-.04, .04, 3), values, c=color,
                                       edgecolors='black', linewidths=.5, s=17, zorder=3)
            for position, case in enumerate(CASES[1:]):
                values = [r[stage] for r in dataset['paired_validation_gaps_pp'] if r['case'] == case]
                x = position+offset
                axes[1, column].bar(x, mean(values), .3, color=color)
                axes[1, column].scatter(x+np.linspace(-.04, .04, 3), values, c=color,
                                       edgecolors='black', linewidths=.5, s=17, zorder=3)
        axes[0, column].set_title(dataset['name']+f"\n{dataset['compact_real_parameters']:,}/{dataset['parent_real_parameters']:,} real parameters", fontsize=11)
        axes[0, column].set_ylim(0, 100)
        axes[0, column].set_xticks(range(3), ['Original', 'Roles preserved', 'Roles free'])
        axes[1, column].set_xticks(range(2), ['Role-preserving mix', 'Free-role mix'])
        axes[1, column].axhline(0, color='black', linewidth=.8)
        for row in (0, 1):
            axes[row, column].set_axisbelow(True)
            axes[row, column].grid(axis='y', alpha=.2)
            axes[row, column].spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('Validation accuracy (%)')
    axes[1, 0].set_ylabel('Original minus mixed accuracy (pp)')
    axes[0, 0].legend(loc='lower left', frameon=False, fontsize=9)
    fig.suptitle('Fixed-scalar-set grouping in physically compact quaternion networks', fontsize=13)
    fig.text(.5, .015, 'Bars: means of 3 paired source seeds. Dots: individual seeds. Previously explored validation splits.\n'
             'Half hidden width retained at two sites; mixing affects all retained quartets at those sites.\n'
             'Different architectures, inputs and savings: 14.5% / 39.4%. Repair attempts: 1125 / 462; actual AMP updates recorded separately.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .95))
    for extension in ('png', 'pdf'):
        fig.savefig(folder/f'grouping_replications.{extension}', dpi=180, bbox_inches='tight')
    result = {'descriptive_only': True, 'datasets': datasets, 'official_test_evaluated': False,
              'source_summary_hashes': {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in paths},
              'script_sha256': file_sha256(Path(__file__)), 'matplotlib': matplotlib.__version__,
              'generated_utc': datetime.now(timezone.utc).isoformat()}
    (output/'cross_dataset_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    lines = ['# Grouping replication across two datasets/architectures', '',
             'Descriptive means across three source seeds; all individual values remain in per-study reports.', '',
             '| Setting | Condition | Native accuracy | After 3 epochs | Final CE |',
             '|---|---|---:|---:|---:|']
    for dataset in datasets:
        for row in dataset['aggregates']:
            lines.append(f"| {dataset['name']} | {row['case']} | {row['before']['accuracy_percent']:.2f}% | {row['after']['accuracy_percent']:.2f}% | {row['after']['cross_entropy']:.6f} |")
    lines += ['', '## Actual budgets', '']
    for dataset in datasets:
        lines.append(f"- {dataset['name']}: {dataset['compact_real_parameters']}compact/{dataset['parent_real_parameters']}parent real trainables, "
                     f"{dataset['whole_model_savings_percent']:.2f}%whole-model savings; attempted updates/model {dataset['attempted_updates_per_model']}; "
                     f"successful updates {dataset['successful_update_range']}.")
    lines += ['', '## Interpretation limits', '',
              'Same retained scalar sets and compact sizes within each source seed/setting. Dataset, architecture, input encoding, '
              'whole-model savings and total repair update budgets differ across settings. Differences in gap magnitude do not isolate '
              'a dataset or quaternion-semantic effect. No pooled significance, equivalence, method novelty or blind-validation claim. '
              'AMP skips are retained; equal attempted updates need not mean equal successful updates.', '',
              '[Figure](figures/grouping_replications.png). No official-test inference.', '']
    (output/'CROSS_DATASET.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'datasets': datasets}, indent=2))


if __name__ == '__main__':
    main()
