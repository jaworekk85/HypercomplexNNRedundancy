from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

def file_sha256(path):
    digest = sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/grouping_statistics_control'
source = json.loads((OUT/'summary.json').read_text())
assert source['status'] == 'completed'
assert source['reported_rows'] == 1512 and source['new_fits'] == 1260
aggregates,counts = [],[]
for dataset in ('cifar10','eurosat'):
    sites = [r for r in source['comparisons'] if r['dataset']==dataset]
    assert len(sites) == 6
    for mode in ('native','bn','standardized'):
        for metric in ('raw_energy','fit_centered_energy'):
            npositive = sum(r['contrasts'][mode+'__'+metric][policy]['increment'] > 0 for r in sites for policy in ('role','free'))
            counts.append({'dataset':dataset,'mode':mode,'metric':metric,'positive_training_increments':npositive,'seed_site_policy_pairs':12})
            for site in (0,1):
                for policy in ('role','free'):
                    values = [r['contrasts'][mode+'__'+metric][policy] for r in sites if r['site']==site]
                    aggregates.append({'dataset':dataset,'site':site,'mode':mode,'metric':metric,'policy':policy,
                        **{k:mean(v[k] for v in values) for k in ('initial','trained','increment')}})
protocol = json.loads((OUT/'protocol.json').read_text())
frozen = {**protocol['source_hashes'],**protocol['input_hashes']}
for path,digest in frozen.items():
    assert file_sha256(ROOT/path) == digest,path
for cell in source['cells']:
    assert file_sha256(ROOT/cell['coefficient_bank']) == cell['coefficient_sha256']
bn_audits = []
for path in sorted(OUT.glob('*/seed_*/bn_model_*/audit.json')):
    audit = json.loads(path.read_text())
    assert audit['trainable_parameters_unchanged']
    assert audit['parameter_hash_before'] == audit['parameter_hash_after']
    assert file_sha256(path.parent/'model.pt') == audit['sha256']
    bn_audits.append(str(path.relative_to(ROOT)))
assert len(bn_audits) == 12
result = {'descriptive_only':True,'source_seed_count_per_dataset':3,'aggregates':aggregates,'counts':counts,
    'reported_rows':source['reported_rows'],'new_fits':source['new_fits'],'frozen_source_input_hashes_checked':len(frozen),
    'coefficient_bank_hashes_checked':len(source['cells']),'bn_model_hashes_checked':len(bn_audits),
    'bn_trainable_parameters_unchanged':True,'verified_utc':datetime.now(timezone.utc).isoformat()}
(OUT/'descriptive_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'counts':counts,'primary':[r for r in aggregates if r['mode']=='standardized' and r['metric']=='fit_centered_energy']},indent=2))
