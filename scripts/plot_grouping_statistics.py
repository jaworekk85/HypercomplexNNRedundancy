"""Descriptive paper figure from completed results; no statistical inference."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from src.data.cifar_cached import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'results/grouping_statistics_control'
    summary = json.loads((output/'summary.json').read_text())
    if summary['status'] != 'completed' or len(summary['comparisons']) != 12:
        raise ValueError('Figure requires the complete two-dataset study')
    figure_dir = output/'figures'
    figure_dir.mkdir(exist_ok=True)
    colors = {'role':'#287d8e','free':'#d67b2b'}
    modes = ['native','bn','standardized']
    fig,axes = plt.subplots(2,2,figsize=(10,6.7),sharey=True)
    for row,dataset in enumerate(['cifar10','eurosat']):
        for column,site in enumerate([0,1]):
            ax = axes[row,column]
            for position,mode in enumerate(modes):
                for policy,offset in [('role',-.17),('free',.17)]:
                    values = [c['contrasts'][mode+'__fit_centered_energy'][policy]['increment']
                              for c in summary['comparisons'] if c['dataset']==dataset and c['site']==site]
                    if len(values) != 3:
                        raise AssertionError('Missing paired source seed')
                    ax.bar(position+offset,np.mean(values),width=.3,color=colors[policy],alpha=.85,zorder=2)
                    ax.scatter(position+offset+np.linspace(-.045,.045,3),values,s=20,color=colors[policy],
                               edgecolors='black',linewidths=.5,zorder=3)
            ax.axhline(0,color='#333333',linewidth=.9,zorder=1)
            ax.set_xticks(range(3),['Native','BN recalibrated','BN + standardized'])
            ax.set_title(('CIFAR-10 / ResNet-20' if dataset=='cifar10' else 'EuroSAT / CNN6')+
                         (' — early consumer' if site==0 else ' — late consumer'),fontsize=10)
            ax.grid(axis='y',alpha=.2,zorder=0)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(labelsize=9)
    fig.suptitle('Training contrast with and without statistics controls',fontsize=13,y=.98)
    fig.supylabel('Trained minus initial grouping penalty\n(centered target-energy normalization)',fontsize=10,x=.015)
    fig.legend(handles=[Patch(color=colors['role'],label='Roles preserved'),Patch(color=colors['free'],label='Roles free')],
               loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
    fig.text(.5,.015,'Bars: mean of 3 source seeds. Dots: individual seeds. Each penalty averages 10 fixed random groupings.\n'
             'Positive values: training increases the mixing penalty. Descriptive comparison; no significance claim.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(.05,.12,1,.95))
    for extension in ('png','pdf'):
        fig.savefig(figure_dir/f'training_increment.{extension}',dpi=180,bbox_inches='tight')
    metadata = {'matplotlib':matplotlib.__version__,'source_summary_sha256':file_sha256(output/'summary.json'),
                'plot_source_sha256':file_sha256(Path(__file__)),'metric':'fit_centered_energy',
                'source_seeds_per_bar':3,'no_significance_or_confidence_interval_claimed':True}
    (figure_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(figure_dir.relative_to(ROOT).as_posix())


if __name__ == '__main__':
    main()
