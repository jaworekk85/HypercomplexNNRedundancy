"""Two-dataset BN and feature-marginal controls for grouping sensitivity."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import gc
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from scripts.run_activation_queue import lock
from scripts.resnet_component_reassembly_pilot import read,write,now,make_model,load_data,checked_pack,Paused
from scripts.grouping_mechanism_study import parent_folder
from src.cnn_training import seed_everything,first_update_audit
from src.compression.grouping_mechanism import kept_units,case_groups
from src.compression.grouping_initialization_control import consumer_targets,fit_cell,policy_summary
from src.compression.grouping_statistics_control import (recalibrate_bn,collect_consumer,standardize_packs,
    add_centered_normalization,parameter_hash)
from src.data.cifar_cached import file_sha256
from src.models.eurosat_cnn import make_fourband
from src.trainable_map_repair import atomic_save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/grouping_statistics_control.json'
NEW = [CONFIG,'scripts/grouping_statistics_control.py','src/compression/grouping_statistics_control.py',
       'tests_torch/test_grouping_statistics_control.py']
MODES = ('native','bn','standardized')
STATES = ('initial','trained')


def factory(dataset,config):
    return make_model(read(ROOT/config['baseline_config'])) if dataset == 'cifar10' else make_fourband(read(ROOT/config['eurosat_config']),'quaternion')


def register(config):
    previous = read(ROOT/config['parent_protocol'])
    euro = read(ROOT/'results/eurosat_cnn/protocol.json')
    sources = {**previous['source_hashes'],**euro['source_hashes']}
    for p,digest in {**sources,**previous['input_hashes']}.items():
        if file_sha256(ROOT/p) != digest:
            raise ValueError(f'Frozen dependency changed: {p}')
    sources.update({p:file_sha256(ROOT/p) for p in NEW})
    inputs = dict(previous['input_hashes'])
    paths = [config['parent_protocol'],'results/eurosat_cnn/protocol.json',config['eurosat_config'],
             '.cache/eurosat/prepared/data.json','.cache/eurosat/prepared/eurosat_nrgb.npz','.cache/eurosat/prepared/split_indices.npz']
    eurodata = read(ROOT/'.cache/eurosat/prepared/data.json')
    if file_sha256(ROOT/paths[-2]) != eurodata['arrays_sha256'] or file_sha256(ROOT/paths[-1]) != eurodata['split_sha256']:
        raise ValueError('EuroSAT data hash mismatch')
    for seed in config['seeds']:
        paths += [f'results/eurosat_cnn/seed_{seed}/quaternion/model.pt',f'results/eurosat_cnn/seed_{seed}/quaternion/metrics.json',
                  config['output']+f'/initial_eurosat_{seed}.pt']
        for index in (0,8):
            folder = ROOT/config['initial_control']/f'seed_{seed}'/f'block_{index}'
            paths += [(folder/(split+'.pt')).relative_to(ROOT).as_posix() for split in ('fit','check_calibration')]
            paths += [(folder/(split+'.json')).relative_to(ROOT).as_posix() for split in ('fit','check_calibration')]
            for state in STATES:
                paths += [(folder/(state+'__'+state)/name).relative_to(ROOT).as_posix() for name in ('result.json','coefficients.pt')]
    inputs.update({p:file_sha256(ROOT/p) for p in paths})
    definition = {'config':config,'source_hashes':sources,'input_hashes':inputs,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda},
        'test_evaluated':False,'validation_evaluated':False}
    output = ROOT/config['output']
    if (output/'protocol.json').exists():
        if read(output/'protocol.json') != definition:
            raise ValueError('Statistics-control protocol changed')
    else:
        write(output/'protocol.json',definition)
    lines = ['# BN/marginal-statistics and second-dataset grouping controls','','## Hypothesis','',config['hypothesis']]
    for name in ('scope','data','bn','standardization','fit','checks','analysis'):
        lines += ['',f'## {name.capitalize()}','',config[name]]
    lines += ['','Prior art: '+', '.join(config['prior_art'])+' (references.bib).','']
    (output/'PROTOCOL.md').write_text('\n'.join(lines),encoding='utf-8')
    return definition


def native_pack(config,dataset,seed,site,state,model,path,images,mean,std,output,stop):
    if dataset == 'cifar10':
        index = (0,8)[site]
        base = config['initial_control'] if state == 'initial' else config['grouping_study']
        folder = ROOT/base/f'seed_{seed}'/f'block_{index}'
    else:
        folder = output/dataset/f'seed_{seed}'/f'site_{site}'/('native_'+state)
    packs = {}
    for split in ('fit','check_calibration'):
        def build(split=split):
            if dataset == 'cifar10':
                raise FileNotFoundError('Missing frozen native activation pack')
            return collect_consumer(model,path,images[split],mean,std,config['collection_batch_size'],stop)
        value = checked_pack(folder/(split+'.pt'),folder/(split+'.json'),build)
        packs[split] = {k:value[k] for k in ('features','targets')}
    return packs


def calibrated_model(model,folder,images,mean,std,config,state,stop,progress):
    path,meta = folder/'model.pt',folder/'audit.json'
    if path.exists():
        audit = read(meta)
        if file_sha256(path) != audit['sha256']:
            raise ValueError('Recalibrated model changed')
        result = deepcopy(model).eval()
        result.load_state_dict(torch.load(path,map_location=mean.device,weights_only=True)['model'])
        if parameter_hash(result) != parameter_hash(model):
            raise AssertionError('Recalibrated trainable weights changed')
        return result
    result,audit = recalibrate_bn(model,images['bn_train'],mean,std,config['bn_batch_size'],stop,
                                 lambda **fields:progress(stage='bn_recalibration',feature_state=state,**fields))
    atomic_save({'model':{n:v.detach().cpu() for n,v in result.state_dict().items()}},path)
    audit['sha256'] = file_sha256(path)
    write(meta,audit)
    return result


def report(config,output,status,synthetic=False):
    cells = [read(p) for p in sorted(output.glob('*/seed_*/site_*/*/result.json'))]
    comparisons = []
    for dataset in config['datasets']:
        for seed in config['seeds']:
            for site in range(len(config['sites'][dataset])):
                available = {c['cell']:c for c in cells if c['dataset']==dataset and c['seed']==seed and c['site']==site}
                if len(available) != 6:
                    continue
                values = {}
                for mode in MODES:
                    for metric in ('raw_energy','fit_centered_energy'):
                        mapped = {}
                        for state in STATES:
                            rows = deepcopy(available[mode+'__'+state]['rows'])
                            if metric == 'fit_centered_energy':
                                for row in rows:
                                    row['check_normalized_mse'] = row['check_centered_normalized_mse']
                            mapped[state] = policy_summary(rows)
                        values[mode+'__'+metric] = {policy:{'initial':mapped['initial'][policy]['delta_vs_original'],
                            'trained':mapped['trained'][policy]['delta_vs_original'],
                            'increment':mapped['trained'][policy]['delta_vs_original']-mapped['initial'][policy]['delta_vs_original']} for policy in ('role','free')}
                comparisons.append({'dataset':dataset,'seed':seed,'site':site,'contrasts':values})
    result = {'status':status,'scientific_results':not synthetic,'cells':cells,'comparisons':comparisons,
        'reported_rows':sum(len(c['rows']) for c in cells),'new_fits':sum(len(c['rows']) for c in cells if not c['reused_predecessor']),
        'test_evaluated':False,'validation_evaluated':False,'updated_utc':now()}
    write(output/'summary.json',result)
    lines = ['# BN/statistics control and EuroSAT replication','',f'Status: {status}.','',
        '## Hypothesis','',config['hypothesis'],'','## Measured primary endpoint','',
        'Mean10draws excess held-out error normalized by target energy centered using FIT channel means.',
        'Standardized condition follows sequential BN recalibration and per-scalar FIT marginal standardization.','',
        '| Dataset | Seed | Site | Policy | Initial delta | Trained delta | Increment |',
        '|---|---:|---:|---|---:|---:|---:|']
    for c in comparisons:
        for policy,v in c['contrasts']['standardized__fit_centered_energy'].items():
            lines.append(f"| {c['dataset']} | {c['seed']} | {c['site']} | {policy} | {v['initial']:.6g} | {v['trained']:.6g} | {v['increment']:.6g} |")
    lines += ['','## Interpretation and limits','',config['scope'],'',config['standardization'],'',config['analysis'],'',
        'All native/BN-only/raw-energy contrasts and individual draws are retained in summary.json/per-cell results.',
        'No significance, equivalence, semantic identifiability or method novelty claimed.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def execute(config,output,data_loader,model_loader,synthetic=False):
    output.mkdir(parents=True,exist_ok=True)
    def stop():
        if (output/'STOP').exists():
            raise Paused()
    def state(**fields):
        write(output/'state.json',{'status':'running','pid':os.getpid(),**fields,'updated_utc':now()})
    try:
        for dataset in config['datasets']:
            images,mean,std = data_loader(dataset)
            for seed in config['seeds']:
                stop()
                models = model_loader(dataset,seed)
                calibrated = {}
                for s,model in models.items():
                    def progress(**fields):
                        state(dataset=dataset,seed=seed,**fields)
                    calibrated[s] = calibrated_model(model,output/dataset/f'seed_{seed}'/('bn_model_'+s),images,mean,std,config,s,stop,progress)
                for site,path in enumerate(config['sites'][dataset]):
                    folder = output/dataset/f'seed_{seed}'/f'site_{site}'
                    state(stage='native_source',dataset=dataset,seed=seed,site=site)
                    if dataset == 'cifar10':
                        assignment = read(ROOT/config['grouping_study']/f'seed_{seed}'/f'block_{(0,8)[site]}'/'assignments.json')
                    else:
                        native = native_pack(config,dataset,seed,site,'trained',models['trained'],path,images,mean,std,output,stop)
                        kept = kept_units(native['fit']['features'])
                        width = models['trained'].get_submodule(path).in_channels
                        assignment = {'kept':kept,'groups':case_groups(kept,width,seed,site,config)}
                        del native
                    assignment['groups'] = {c:[list(q) for q in groups] for c,groups in assignment['groups'].items() if c == 'original' or '_100_' in c}
                    assignment['source_width'] = models['trained'].get_submodule(path).in_channels
                    if (folder/'assignments.json').exists() and read(folder/'assignments.json') != assignment:
                        raise ValueError('Assignments changed')
                    write(folder/'assignments.json',assignment)
                    for mode in MODES:
                        for s in STATES:
                            stop()
                            name = mode+'__'+s
                            target = folder/name
                            if (target/'result.json').exists():
                                old = read(target/'result.json')
                                if file_sha256(ROOT/old['coefficient_bank']) != old['coefficient_sha256']:
                                    raise ValueError('Completed coefficients changed')
                                continue
                            state(stage='source_features',dataset=dataset,seed=seed,site=site,cell=name)
                            if mode == 'native':
                                packs = native_pack(config,dataset,seed,site,s,models[s],path,images,mean,std,output,stop)
                            else:
                                cache = folder/('bn_features_'+s)
                                packs = {split:checked_pack(cache/(split+'.pt'),cache/(split+'.json'),
                                    lambda split=split:collect_consumer(calibrated[s],path,images[split],mean,std,config['collection_batch_size'],stop)) for split in ('fit','check_calibration')}
                            consumer = models[s].get_submodule(path)
                            scaling = None
                            if mode == 'standardized':
                                packs,scaling = standardize_packs(packs)
                                targets = consumer_targets(consumer,packs,mean.device,config['collection_batch_size'],stop)
                            else:
                                targets = {split:p['targets'] for split,p in packs.items()}
                            reused = dataset == 'cifar10' and mode == 'native' and not synthetic
                            if reused:
                                previous = ROOT/config['initial_control']/f'seed_{seed}'/f'block_{(0,8)[site]}'/(s+'__'+s)
                                payload = read(previous/'result.json')
                                rows,controls = payload['rows'],payload['controls']
                                bank = previous/'coefficients.pt'
                                digest = payload['coefficients_sha256']
                                if file_sha256(bank) != digest:
                                    raise ValueError('Predecessor bank changed')
                            else:
                                shape = (4,consumer.out_channels,len(assignment['kept']),3,3)
                                def fitting(**fields):
                                    state(stage='ridge',dataset=dataset,seed=seed,site=site,cell=name,**fields)
                                rows,kernels,controls = fit_cell(packs,targets,assignment,shape,config['ridge_alpha'],fitting,stop)
                                bank = target/'coefficients.pt'
                                atomic_save({'cases':[r['case'] for r in rows],'kernels':kernels},bank)
                                saved = torch.load(bank,map_location='cpu',weights_only=True)
                                if any(not torch.equal(a,b) for a,b in zip(kernels,saved['kernels'])):
                                    raise AssertionError('Coefficient reload mismatch')
                                digest = file_sha256(bank)
                                del kernels,saved
                            rows,normalization = add_centered_normalization(rows,targets)
                            result = {'dataset':dataset,'seed':seed,'site':site,'cell':name,'rows':rows,'controls':controls,
                                'normalization':normalization,'feature_scaling':scaling,'reused_predecessor':reused,
                                'coefficient_bank':bank.relative_to(ROOT).as_posix(),'coefficient_sha256':digest}
                            write(target/'result.json',result)
                            print(f'CELL {dataset} seed{seed} site{site} {name}: {len(rows)}rows reused={reused}',flush=True)
                            report(config,output,'running',synthetic)
                            del packs,targets
                    report(config,output,'running',synthetic)
                del models,calibrated
                gc.collect()
                if mean.device.type == 'cuda':
                    torch.cuda.empty_cache()
            del images
        result = report(config,output,'completed',synthetic)
        expected = sum(len(config['sites'][d]) for d in config['datasets'])*len(config['seeds'])*6*(1+2*config['random_repeats'])
        if result['reported_rows'] != expected:
            raise AssertionError('Incomplete statistics control')
        for c in result['cells']:
            if file_sha256(ROOT/c['coefficient_bank']) != c['coefficient_sha256']:
                raise AssertionError('Final coefficient-bank hash mismatch')
        write(output/'completion_audit.json',{'status':'passed','reported_rows':result['reported_rows'],
            'new_fits':result['new_fits'],'coefficient_bank_hashes_checked':len(result['cells']),'updated_utc':now()})
        write(output/'state.json',{'status':'completed','reported_rows':result['reported_rows'],'new_fits':result['new_fits'],'updated_utc':now()})
        return result
    except Paused:
        report(config,output,'paused',synthetic)
        write(output/'state.json',{'status':'paused','updated_utc':now()})
        return None
    except BaseException as error:
        report(config,output,'failed',synthetic)
        write(output/'state.json',{'status':'failed','error':f'{type(error).__name__}: {error}','updated_utc':now()})
        raise


def data_for(dataset,config,output):
    device = torch.device(config['device'])
    generator = torch.Generator().manual_seed(config['bn_subset_seed']+(1000 if dataset == 'eurosat' else 0))
    if dataset == 'cifar10':
        tensors,unused,mean,std = load_data(read(ROOT/config['baseline_config']),device)
        indices = torch.randperm(len(tensors['train']),generator=generator)[:config['bn_train_images']]
        images = {'bn_train':tensors['train'][indices.to(device)],**{s:tensors[s] for s in ('fit','check_calibration')}}
    else:
        cfg = read(ROOT/config['eurosat_config'])
        base = ROOT/cfg['data']
        with np.load(base/'eurosat_nrgb.npz') as archive:
            source = archive['images']
        with np.load(base/'split_indices.npz') as split:
            train,cal = split['train'],split['calibration']
            if len(set(train.tolist()).intersection(cal.tolist())) or len(cal) != 1000:
                raise ValueError('Invalid EuroSAT train/calibration split')
        indices = torch.randperm(len(train),generator=generator)[:config['bn_train_images']]
        ids = {'bn_train':train[indices.numpy()],'fit':cal[::2],'check_calibration':cal[1::2]}
        images = {s:torch.as_tensor(source[i].astype(np.int16),device=device) for s,i in ids.items()}
        meta = read(base/'data.json')['statistics']
        mean,std = [torch.tensor(meta[k],device=device).reshape(1,4,1,1) for k in ('mean','std')]
    write(output/dataset/'bn_train_subset.json',{'train_split_positions':indices.tolist(),'images':len(indices),
        'label_free':True,'selection_seed':config['bn_subset_seed']+(1000 if dataset == 'eurosat' else 0)})
    return images,mean,std


def model_pair(dataset,seed,config):
    device = torch.device(config['device'])
    pair = {s:factory(dataset,config).to(device).eval() for s in STATES}
    initial = ROOT/(config['initial_control']+f'/initial_seed_{seed}.pt') if dataset == 'cifar10' else ROOT/config['output']/f'initial_eurosat_{seed}.pt'
    trained = parent_folder(seed)/'model.pt' if dataset == 'cifar10' else ROOT/f'results/eurosat_cnn/seed_{seed}/quaternion/model.pt'
    for s,path in [('initial',initial),('trained',trained)]:
        pair[s].load_state_dict(torch.load(path,map_location=device,weights_only=True)['model'])
    return pair


def smoke(config):
    tiny = deepcopy(config)
    tiny.update(datasets=['toy'],seeds=[19],sites={'toy':['blocks.0.conv2','blocks.8.conv2']},random_repeats=1,
                device='cpu',collection_batch_size=2,bn_batch_size=4,bn_train_images=8)
    baseline = read(ROOT/config['baseline_config'])
    baseline.update(quaternion_widths=[8,8,8],classes=2)
    seed_everything(19)
    base = make_model(baseline).eval()
    trained = deepcopy(base)
    with torch.no_grad():
        trained.blocks[0].conv1.kernel.mul_(1.2)
    images = {s:torch.randint(256,(n,3,8,8),dtype=torch.uint8) for s,n in [('bn_train',8),('fit',4),('check_calibration',4)]}
    output = ROOT/'.cache'/('statistics_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    args = (tiny,output,lambda d:(images,torch.zeros(1,3,1,1),torch.ones(1,3,1,1)),
            lambda d,s:{'initial':deepcopy(base),'trained':deepcopy(trained)},True)
    result = execute(*args)
    replay = execute(*args)
    if result['cells'] != replay['cells'] or result['comparisons'] != replay['comparisons']:
        raise AssertionError('Completed statistics-study resume changed results')
    return {'status':'passed','output':output.relative_to(ROOT).as_posix(),'rows':result['reported_rows']}


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc':now(),'scientific_results':False}
    try:
        tests = subprocess.run([sys.executable,'-B','-m','pytest','-q','tests_torch/test_grouping_statistics_control.py'],cwd=ROOT,capture_output=True,text=True)
        record['tests'] = {'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr}
        print(tests.stdout,flush=True)
        if tests.returncode:
            raise RuntimeError('Statistics-control tests failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = smoke(config)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            for seed in config['seeds']:
                seed_everything(seed)
                model = factory('eurosat',config).eval()
                path = output/f'initial_eurosat_{seed}.pt'
                if path.exists():
                    saved = torch.load(path,map_location='cpu',weights_only=True)['model']
                    if any(not torch.equal(v,saved[k]) for k,v in model.state_dict().items()):
                        raise ValueError('EuroSAT initialization changed')
                else:
                    atomic_save({'model':model.state_dict()},path)
            probe = deepcopy(model).cuda().train()
            record['gpu_fourband_update_audit'] = first_update_audit(probe,torch.randn(8,4,32,32,device='cuda'),
                torch.arange(8,device='cuda')%10,torch.optim.SGD(probe.parameters(),lr=.001))
            del probe,model
            definition = register(config)
            record.update(status='passed',source_hashes=definition['source_hashes'],updated_utc=now())
            write(output/'readiness.json',record)
            write(output/'state.json',{'status':'ready_not_started','updated_utc':now()})
    except BaseException as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        history = read(output/'development_checks.json') if (output/'development_checks.json').exists() else []
        history.append(record)
        write(output/'development_checks.json',history)
    print('READY: statistics controls and EuroSAT replication registered',flush=True)


def run(config):
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        output = ROOT/config['output']
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Readiness mismatch')
        result = execute(config,output,lambda d:data_for(d,config,output),lambda d,s:model_pair(d,s,config))
        if result is not None:
            register(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    config = read(ROOT/CONFIG)
    torch.set_num_threads(config['cpu_threads'])
    run(config) if args.run else prepare(config)
