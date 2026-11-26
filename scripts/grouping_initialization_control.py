"""Run the paired step0/trained and crossed grouping diagnostic."""
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
from scripts.resnet_component_reassembly_pilot import (read,write,now,make_model,save_model,load_model,
    load_data,collect_source,checked_pack,Paused)
from scripts.grouping_mechanism_study import parent_folder
from src.cnn_training import first_update_audit,seed_everything
from src.compression.grouping_initialization_control import consumer_targets,fit_cell,crossed_summary
from src.compression.grouping_mechanism import case_groups,dense_permutation_check
from src.data.cifar_cached import file_sha256
from src.trainable_map_repair import atomic_save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/grouping_initialization_control.json'
NEW = [CONFIG,'scripts/grouping_initialization_control.py','src/compression/grouping_initialization_control.py',
       'tests_torch/test_grouping_initialization_control.py']
CELLS = [('initial','initial'),('initial','trained'),('trained','initial'),('trained','trained')]


def register(config):
    previous = read(ROOT/config['parent_protocol'])
    for p,digest in {**previous['source_hashes'],**previous['input_hashes']}.items():
        if file_sha256(ROOT/p) != digest:
            raise ValueError(f'Frozen dependency changed: {p}')
    inputs = dict(previous['input_hashes'])
    inputs[config['parent_protocol']] = file_sha256(ROOT/config['parent_protocol'])
    for seed in config['seeds']:
        path = ROOT/config['output']/f'initial_seed_{seed}.pt'
        inputs[path.relative_to(ROOT).as_posix()] = file_sha256(path)
        for index in config['blocks']:
            folder = ROOT/config['previous_output']/f'seed_{seed}'/f'block_{index}'
            names = ['assignments.json','controls.json','fit.pt','fit.json','check_calibration.pt','check_calibration.json']
            for case in read(folder/'assignments.json')['groups']:
                if case == 'original' or '_100_' in case:
                    names.append(case+'/result.json')
            for name in names:
                p = folder/name
                inputs[p.relative_to(ROOT).as_posix()] = file_sha256(p)
            for split in ('fit','check_calibration'):
                if file_sha256(folder/(split+'.pt')) != read(folder/(split+'.json'))['sha256']:
                    raise ValueError('Predecessor activation pack changed')
    definition = {'config':config,'source_hashes':{**previous['source_hashes'],**{p:file_sha256(ROOT/p) for p in NEW}},
        'input_hashes':inputs,'environment':{'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'cuda':torch.version.cuda},'test_evaluated':False,'validation_evaluated':False}
    output = ROOT/config['output']
    if (output/'protocol.json').exists():
        if read(output/'protocol.json') != definition:
            raise ValueError('Registered initialization control changed')
    else:
        write(output/'protocol.json',definition)
    lines = ['# Initialization versus training grouping control','','## Hypothesis','',config['hypothesis']]
    for name in ('design','initialization','crossing','fit','controls','analysis'):
        lines += ['',f'## {name.capitalize()}','',config[name]]
    lines += ['','Prior art: '+', '.join(config['prior_art'])+' (references.bib).','']
    (output/'PROTOCOL.md').write_text('\n'.join(lines),encoding='utf-8')
    return definition


def report(config,output,status,synthetic=False):
    cells = [read(p) for p in sorted(output.glob('seed_*/block_*/*/result.json'))]
    sites = [read(p) for p in sorted(output.glob('seed_*/block_*/comparison.json'))]
    write(output/'summary.json',{'status':status,'scientific_results':not synthetic,'cells':cells,'sites':sites,
        'ridge_fits':sum(len(c['rows']) for c in cells),'test_evaluated':False,'validation_evaluated':False,
        'statistical_significance_claimed':False,'updated_utc':now()})
    lines = ['# Grouping before and after training','',f'Status: {status}.','','## Hypothesis','',config['hypothesis'],
        '','## Measured normalized grouping penalties','',
        'Delta = mean held-out normalized MSE of10fully mixed draws minus original-group normalized MSE.',
        'Positive training increment means a larger penalty after training.','',
        '| Seed | Block | Policy | Initial delta | Trained delta | Training increment | Feature-only increment | Consumer-only increment | Interaction |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for site in sites:
        for policy,c in site['summary']['contrasts'].items():
            lines.append(f"| {site['seed']} | {site['block']} | {policy} | {c['initial_penalty']:.6g} | {c['trained_penalty']:.6g} | {c['training_increment']:.6g} | {c['feature_increment_at_initial_consumer']:.6g} | {c['consumer_increment_at_initial_features']:.6g} | {c['interaction']:.6g} |")
    lines += ['','## Interpretation and limits','',config['design'],'',config['initialization'],'',config['analysis'],'',
        'Initialization uses the same Hamilton architecture, not a random algebra. This is an exact-feature diagnostic;',
        'it establishes no deployed compact architecture savings or random-model classification comparison.',
        'Per-cell rows retain all draws, target energies, raw/normalized MSE, ridge penalties and replay controls.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return read(output/'summary.json')


def execute(config,output,load_site,synthetic=False):
    output.mkdir(parents=True,exist_ok=True)
    def stop():
        if (output/'STOP').exists():
            raise Paused()
    def state(**fields):
        write(output/'state.json',{'status':'running','pid':os.getpid(),**fields,'updated_utc':now()})
    try:
        for seed in config['seeds']:
            for index in config['blocks']:
                stop()
                folder = output/f'seed_{seed}'/f'block_{index}'
                if (folder/'comparison.json').exists():
                    for feature,consumer in CELLS:
                        p = folder/(feature+'__'+consumer)
                        row = read(p/'result.json')
                        if file_sha256(p/'coefficients.pt') != row['coefficients_sha256']:
                            raise ValueError('Completed coefficient bank changed')
                    continue
                state(stage='source_activations',seed=seed,block=index)
                site = load_site(seed,index,stop)
                assignment = site['assignment']
                assignment['groups'] = {c:qs for c,qs in assignment['groups'].items() if c == 'original' or '_100_' in c}
                assignment['source_width'] = site['consumers']['trained'].in_channels
                all_rows,lambdas = {},{}
                for feature_state,consumer_state in CELLS:
                    stop()
                    name = feature_state+'__'+consumer_state
                    target = folder/name
                    if (target/'result.json').exists():
                        result = read(target/'result.json')
                        if file_sha256(target/'coefficients.pt') != result['coefficients_sha256']:
                            raise ValueError('Coefficient bank changed')
                    else:
                        packs = site['packs'][feature_state]
                        consumer = site['consumers'][consumer_state]
                        state(stage='targets',seed=seed,block=index,cell=name)
                        # Use original exact targets for diagonal cells; cross cells apply full-width consumer.
                        if feature_state == consumer_state:
                            targets = {s:p['targets'] for s,p in packs.items()}
                        else:
                            targets = consumer_targets(consumer,packs,config['device'],config['collection_batch_size'],stop)
                        shape = (4,consumer.out_channels,len(assignment['kept']),3,3)
                        def progress(**fields):
                            state(stage='ridge',seed=seed,block=index,cell=name,**fields)
                        rows,kernels,controls = fit_cell(packs,targets,assignment,shape,config['ridge_alpha'],progress,stop)
                        # Numerical dense control uses the correct full source consumer and exact features.
                        from types import SimpleNamespace
                        fake_block = SimpleNamespace(conv1=SimpleNamespace(out_channels=assignment['source_width']),conv2=consumer)
                        case = 'free_100_0'
                        controls['dense_inverse_permutation_max_abs'] = dense_permutation_check(fake_block,packs['fit']['features'],assignment['kept'],assignment['groups'][case])
                        replay_count = 0
                        if name == 'trained__trained' and not synthetic:
                            for row in rows:
                                previous = read(ROOT/config['previous_output']/f'seed_{seed}'/f'block_{index}'/row['case']/'result.json')['ideal']
                                for field in ('fit_mse','check_mse'):
                                    if abs(row[field]-previous[field]) > 1e-7*max(1.,previous[field]):
                                        raise AssertionError(f'Predecessor replay failed: {seed}/{index}/{row["case"]}/{field}')
                                replay_count += 1
                        payload = {'cases':[r['case'] for r in rows],'kernels':kernels}
                        atomic_save(payload,target/'coefficients.pt')
                        loaded = torch.load(target/'coefficients.pt',map_location='cpu',weights_only=True)
                        if loaded['cases'] != payload['cases'] or any(not torch.equal(a,b) for a,b in zip(kernels,loaded['kernels'])):
                            raise AssertionError('Serialized coefficient bank changed')
                        result = {'seed':seed,'block':index,'cell':name,'rows':rows,'controls':controls,
                            'predecessor_rows_replayed':replay_count,'coefficient_bank_reload_exact':True,
                            'coefficients_sha256':file_sha256(target/'coefficients.pt')}
                        write(target/'result.json',result)
                        print(f'CELL seed{seed} block{index} {name}: {len(rows)}fits, replay={replay_count}',flush=True)
                        del kernels,loaded,payload,targets
                    all_rows[name] = result['rows']
                    lambdas[name] = result['controls']['lambda']
                    report(config,output,'running',synthetic)
                for feature in ('initial','trained'):
                    if lambdas[feature+'__initial'] != lambdas[feature+'__trained']:
                        raise AssertionError('Lambda changed with consumer weights')
                comparison = {'seed':seed,'block':index,'summary':crossed_summary(all_rows)}
                write(folder/'comparison.json',comparison)
                for policy,c in comparison['summary']['contrasts'].items():
                    print(f"COMPARE seed{seed} block{index} {policy}: initialDelta={c['initial_penalty']:.6g} trainedDelta={c['trained_penalty']:.6g} increment={c['training_increment']:.6g}",flush=True)
                del site
                gc.collect()
        result = report(config,output,'completed',synthetic)
        expected = len(config['seeds'])*len(config['blocks'])*4*(1+2*config['random_repeats'])
        if result['ridge_fits'] != expected:
            raise AssertionError('Incomplete initialization diagnostic')
        replayed = sum(c['predecessor_rows_replayed'] for c in result['cells'])
        write(output/'completion_audit.json',{'status':'passed','ridge_fits':expected,'predecessor_rows_replayed':replayed,
            'coefficient_banks':len(result['cells']),'updated_utc':now()})
        write(output/'state.json',{'status':'completed','ridge_fits':expected,'sites':len(result['sites']),'updated_utc':now()})
        return result
    except Paused:
        report(config,output,'paused',synthetic)
        write(output/'state.json',{'status':'paused','updated_utc':now()})
        return None
    except BaseException as error:
        report(config,output,'failed',synthetic)
        write(output/'state.json',{'status':'failed','error':f'{type(error).__name__}: {error}','updated_utc':now()})
        raise


def smoke(config):
    from src.models.cifar_resnet import BasicBlock
    tiny = deepcopy(config)
    tiny.update(seeds=[19],blocks=[0],random_repeats=1,device='cpu',collection_batch_size=2)
    seed_everything(19)
    initial = BasicBlock('quaternion',2,8).eval()
    trained = deepcopy(initial)
    with torch.no_grad():
        trained.conv1.kernel.mul_(1.3)
        trained.conv2.kernel.add_(.02)
    x = torch.randn(8,8,5,5)
    def pack(model):
        with torch.no_grad():
            h = torch.relu(model.bn1(model.conv1(x)))
            y = model.conv2(h)
        return {s:{'features':h[ids],'targets':y[ids]} for s,ids in [('fit',slice(0,4)),('check_calibration',slice(4,8))]}
    packs = {'initial':pack(initial),'trained':pack(trained)}
    assignment = {'kept':[0,1,2,3],'groups':case_groups([0,1,2,3],8,19,0,{**tiny,'shuffle_seed_offset':210000})}
    def loader(seed,index,stop):
        return {'packs':packs,'consumers':{'initial':initial.conv2,'trained':trained.conv2},'assignment':deepcopy(assignment)}
    output = ROOT/'.cache'/('grouping_initial_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    result = execute(tiny,output,loader,True)
    replay = execute(tiny,output,loader,True)
    if result['cells'] != replay['cells'] or result['sites'] != replay['sites']:
        raise AssertionError('Completed-cell resume changed results')
    return {'status':'passed','output':output.relative_to(ROOT).as_posix(),'ridge_fits':result['ridge_fits']}


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc':now(),'scientific_results':False}
    try:
        tests = subprocess.run([sys.executable,'-B','-m','pytest','-q','tests_torch/test_grouping_initialization_control.py'],cwd=ROOT,capture_output=True,text=True)
        record['tests'] = {'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr}
        print(tests.stdout,flush=True)
        if tests.returncode:
            raise RuntimeError('Initialization-control tests failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = smoke(config)
            baseline = read(ROOT/config['baseline_config'])
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            for seed in config['seeds']:
                seed_everything(seed)
                model = make_model(baseline).eval()
                path = output/f'initial_seed_{seed}.pt'
                if path.exists():
                    replica = load_model(path,baseline,'cpu')
                    if any(not torch.equal(v,replica.state_dict()[n]) for n,v in model.state_dict().items()):
                        raise ValueError('Saved initialization differs from recipe')
                else:
                    save_model(model,path)
            probe = deepcopy(model).cuda().train()
            record['gpu_discarded_copy_update_audit'] = first_update_audit(probe,torch.randn(8,3,32,32,device='cuda'),
                torch.arange(8,device='cuda')%baseline['classes'],torch.optim.SGD(probe.parameters(),lr=.001))
            del model,probe
            torch.cuda.empty_cache()
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
    print('READY: initialization control registered before fitting',flush=True)


def run(config):
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        output = ROOT/config['output']
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Readiness mismatch')
        device = torch.device(config['device'])
        baseline = read(ROOT/config['baseline_config'])
        tensors,unused_labels,mean,std = load_data(baseline,device)
        del unused_labels
        cache = {}
        def loader(seed,index,stop):
            if cache.get('seed') != seed:
                cache.clear()
                gc.collect()
                cache.update(seed=seed,initial=load_model(output/f'initial_seed_{seed}.pt',baseline,device))
                model = make_model(baseline).to(device).eval()
                model.load_state_dict(torch.load(parent_folder(seed)/'model.pt',map_location=device,weights_only=True)['model'])
                cache['trained'] = model
            previous = ROOT/config['previous_output']/f'seed_{seed}'/f'block_{index}'
            folder = output/f'seed_{seed}'/f'block_{index}'
            packs = {'trained':{},'initial':{}}
            def missing_predecessor():
                raise FileNotFoundError('Missing frozen predecessor activation cache')
            for split in ('fit','check_calibration'):
                packs['trained'][split] = checked_pack(previous/(split+'.pt'),previous/(split+'.json'),missing_predecessor)
                packs['initial'][split] = checked_pack(folder/(split+'.pt'),folder/(split+'.json'),
                    lambda split=split:collect_source(cache['initial'],index,tensors[split],mean,std,config['collection_batch_size'],stop))
            return {'packs':packs,'consumers':{s:cache[s].blocks[index].conv2 for s in ('initial','trained')},
                    'assignment':read(previous/'assignments.json')}
        result = execute(config,output,loader)
        if result is not None:
            register(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    config = read(ROOT/CONFIG)
    torch.set_num_threads(config['cpu_threads'])
    run(config) if args.run else prepare(config)
