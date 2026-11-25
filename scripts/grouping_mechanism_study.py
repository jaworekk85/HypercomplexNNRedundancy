"""Prepare and run fixed-feature grouping, producer-realization and repair studies."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.run_activation_queue import lock
from scripts.resnet_component_reassembly_pilot import (read,write,now,make_model,save_model,load_model,
    load_data,collect_source,checked_pack,metrics,assert_replay,Paused)
from src.cnn_training import evaluate,first_update_audit,prepare_batch,seed_everything
from src.compression.component_reassembly import packing_indices
from src.compression.grouping_mechanism import (kept_units,case_groups,raw,permute_moments,q_moments,solve,
    relative_permutation,dense_permutation_check,physical_fit,fit_producer)
from src.compression.resnet_reassembly import compile_block,original_quartets,local_mse
from src.data.cifar_cached import file_sha256
from src.trainable_map_repair import atomic_save,repair

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/grouping_mechanism_study.json'
NEW = [CONFIG,'scripts/grouping_mechanism_study.py','src/compression/grouping_mechanism.py',
       'tests_torch/test_grouping_mechanism.py']


def parent_folder(seed):
    return ROOT/('results/resnet20_baselines' if seed == 2026 else f'results/resnet20_replication_baselines/seed_{seed}')/'quaternion'


def register(config):
    previous = read(ROOT/config['parent_protocol'])
    inherited = {**previous['source_hashes'],**previous['input_hashes']}
    for path,digest in inherited.items():
        if file_sha256(ROOT/path) != digest:
            raise ValueError(f'Frozen dependency changed: {path}')
    inputs = dict(previous['input_hashes'])
    for seed in config['seeds']:
        folder = parent_folder(seed)
        info = read(folder/'metrics.json')
        if file_sha256(folder/'model.pt') != info['checkpoint_sha256']:
            raise ValueError(f'Parent changed: {seed}')
        for name in ('model.pt','metrics.json'):
            path = (folder/name).relative_to(ROOT).as_posix()
            inputs[path] = file_sha256(ROOT/path)
    inputs[config['parent_protocol']] = file_sha256(ROOT/config['parent_protocol'])
    definition = {'config':config,'source_hashes':{**previous['source_hashes'],**{p:file_sha256(ROOT/p) for p in NEW}},
                  'input_hashes':inputs,'environment':{'python':platform.python_version(),'torch':torch.__version__,
                  'numpy':np.__version__,'cuda':torch.version.cuda},'test_evaluated':False}
    output = ROOT/config['output']
    path = output/'protocol.json'
    if path.exists() and read(path) != definition:
        raise ValueError('Grouping study protocol changed')
    if not path.exists():
        write(path,definition)
    lines = ['# Fixed-feature grouping mechanism study','','## Hypotheses','']+config['hypotheses']
    for key in ('scope','groups','ideal','physical','full','interpretation','resume'):
        lines += ['',f'## {key.capitalize()}','',config[key]]
    lines += ['','Prior art: '+', '.join(config['prior_art'])+'; canonical references.bib.','',
              'Numerical settings, fixed seeds and exact hashes: protocol.json.','']
    (output/'PROTOCOL.md').write_text('\n'.join(lines),encoding='utf-8')
    return definition


def verified(path,metadata):
    if file_sha256(path) != metadata['sha256']:
        raise ValueError(f'Artifact changed: {path}')


def save_block(block,groups,path):
    atomic_save({'model':{n:v.detach().cpu() for n,v in block.state_dict().items()},'quartets':[list(q) for q in groups]},path)
    return {'sha256':file_sha256(path)}


def restore_block(source,path,metadata,device):
    verified(path,metadata)
    payload = torch.load(path,map_location=device,weights_only=True)
    block,_ = compile_block(source,[tuple(q) for q in payload['quartets']])
    block.load_state_dict(payload['model'])
    return block.eval()


def report(config,output,status,synthetic=False):
    local = [read(p) for p in sorted(output.glob('seed_*/block_*/*/result.json'))]
    full = [read(p) for p in sorted(output.glob('seed_*/models/*/result.json'))]
    summary = {'status':status,'scientific_results':not synthetic,'local_results':local,'results':full,
               'test_evaluated':False,'statistical_significance_or_equivalence_claimed':False,'updated_utc':now()}
    write(output/'summary.json',summary)
    lines = ['# Fixed-feature grouping study','',f'Status: {status}.','',
             '## Hypotheses','']+config['hypotheses']+['','## Measured full-model results','',
             '| Seed | Case | Real trainables | Validation before (%) | After (%) | CE after |',
             '|---|---|---:|---:|---:|---:|']
    for row in full:
        lines.append(f"| {row['seed']} | {row['case']} | {row['real_trainable_parameters']} | {100*row['before']['validation']['accuracy']:.2f} | {100*row['after']['validation']['accuracy']:.2f} | {row['after']['validation']['cross_entropy']:.6f} |")
    lines += ['','## Measured local results','',
              '| Seed | Block | Case | Mixed | Ideal check MSE | Physical check MSE | Data-fit check MSE |',
              '|---|---:|---|---:|---:|---:|---:|']
    for row in local:
        value = row.get('datafit',{}).get('fit',{}).get('check_mse')
        lines.append(f"| {row['seed']} | {row['block']} | {row['case']} | {row['description']['mixed_quartets']} | {row['ideal']['check_mse']:.6g} | {row['physical']['check_mse']:.6g} | {value if value is not None else '-'} |")
    lines += ['','## Interpretation and limits','',config['scope'],'',config['interpretation'],'',
              'All draws are reported; full-network draw0 is predetermined, never best-of-draws.',
              'The earlier pilot continuation gate remains failed. No automatic follow-up.','',
              'Prior art: '+', '.join(config['prior_art'])+' (references.bib).','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def local_studies(config,baseline,source,tensors,targets,mean,std,seed,output,state,stop):
    device = mean.device
    for index in config['blocks']:
        stop()
        folder = output/f'seed_{seed}'/f'block_{index}'
        original = source.blocks[index]
        expected = 3+2*config['random_repeats']
        complete = list(folder.glob('*/result.json'))
        if len(complete) == expected:
            for p in complete:
                row = read(p)
                verified(p.parent/'block.pt',row['checkpoint'])
                if 'datafit' in row:
                    verified(p.parent/'datafit_block.pt',row['datafit']['checkpoint'])
            continue
        packs = {}
        for split in ('fit','check_calibration'):
            state(stage='source_activations',seed=seed,block=index,split=split)
            packs[split] = checked_pack(folder/f'{split}.pt',folder/f'{split}.json',
                lambda split=split:collect_source(source,index,tensors[split],mean,std,config['collection_batch_size'],stop))
        kept = kept_units(packs['fit']['features'])
        width = original.conv1.out_channels
        groups = case_groups(kept,width,seed,index,config)
        assignment = {'kept':kept,'groups':{c:[list(q) for q in qs] for c,qs in groups.items()}}
        if (folder/'assignments.json').exists() and read(folder/'assignments.json') != assignment:
            raise ValueError('Fixed assignments changed')
        write(folder/'assignments.json',assignment)
        original_ids = packing_indices([original_quartets(width)[i] for i in kept],width)
        def build_moments():
            return {s:raw(p['features'][:,original_ids],p['targets']) for s,p in packs.items()}
        state(stage='exact_feature_moments',seed=seed,block=index)
        moments = checked_pack(folder/'moments.pt',folder/'moments.json',build_moments)
        gram,_ = q_moments(moments['fit'],original.conv2.out_channels)
        penalty = config['ridge_alpha']*float(gram.diag().mean())
        if penalty <= 0:
            raise ValueError('Degenerate fixed-feature Gram')
        shape = (4,original.conv2.out_channels,len(kept),3,3)
        # Coherently permuting entire units must preserve the Hamilton optimum.
        reordered = [original_quartets(width)[i] for i in reversed(kept)]
        permutation = relative_permutation(reordered,kept,width)
        _,ref = solve(moments['fit'],moments['check_calibration'],shape,penalty)
        _,replay = solve(permute_moments(moments['fit'],permutation),permute_moments(moments['check_calibration'],permutation),shape,penalty)
        if abs(ref['regularized_fit_objective']-replay['regularized_fit_objective']) > 1e-8*max(1.,abs(ref['regularized_fit_objective'])):
            raise AssertionError('Whole-unit permutation control failed')
        write(folder/'controls.json',{'unit_reorder_original':ref,'unit_reorder_replayed':replay,'lambda':penalty})
        for case,quartets in groups.items():
            stop()
            target = folder/case
            if (target/'result.json').exists():
                old = read(target/'result.json')
                verified(target/'block.pt',old['checkpoint'])
                if 'datafit' in old:
                    verified(target/'datafit_block.pt',old['datafit']['checkpoint'])
                continue
            start = time.perf_counter()
            state(stage='ideal_grouping_fit',seed=seed,block=index,case=case)
            permutation = relative_permutation(quartets,kept,width)
            ideal_moments = {s:permute_moments(m,permutation) for s,m in moments.items()}
            _,ideal = solve(ideal_moments['fit'],ideal_moments['check_calibration'],shape,penalty)
            inverse_check = dense_permutation_check(original,packs['fit']['features'],kept,quartets)
            block,description = compile_block(original,quartets)
            ids = packing_indices(quartets,width)
            selected = {s:{**p,'selected_features':p['features'][:,ids]} for s,p in packs.items()}
            state(stage='physical_producer_fit',seed=seed,block=index,case=case)
            physical = physical_fit(block,selected,penalty,device,config['collection_batch_size'],stop)
            actual = local_mse(block,packs['check_calibration'],device,config['collection_batch_size'])
            if abs(actual-physical['check_mse']) > 1e-6*max(1.,physical['check_mse']):
                raise AssertionError('Physical float32 MSE differs from float64 moment fit')
            if case == 'original' and abs(ideal['check_mse']-physical['check_mse']) > 1e-8*max(1.,ideal['check_mse']):
                raise AssertionError('Original producer realization is not exact')
            checkpoint = save_block(block,quartets,target/'block.pt')
            restored = restore_block(original,target/'block.pt',checkpoint,device)
            if abs(local_mse(restored,packs['check_calibration'],device,config['collection_batch_size'])-actual) > 1e-10*max(1.,actual):
                raise AssertionError('Local artifact replay failed')
            row = {'seed':seed,'block':index,'case':case,'kept_units':kept,'description':description,
                   'ideal':ideal,'physical':physical,'physical_float32_check_mse':actual,
                   'dense_inverse_permutation_max_abs':inverse_check,'checkpoint':checkpoint}
            if case in ('role_100_0','free_100_0'):
                state(stage='data_aware_producer',seed=seed,block=index,case=case)
                fitted,audit = fit_producer(block,selected['fit'],config['producer_fit'],seed+index,device,stop)
                fit = physical_fit(fitted,selected,penalty,device,config['collection_batch_size'],stop)
                meta = save_block(fitted,quartets,target/'datafit_block.pt')
                restored_fit = restore_block(original,target/'datafit_block.pt',meta,device)
                exact = local_mse(restored_fit,packs['check_calibration'],device,config['collection_batch_size'])
                if abs(exact-fit['check_mse']) > 1e-6*max(1.,fit['check_mse']):
                    raise AssertionError('Data-fit producer replay failed')
                row['datafit'] = {'audit':audit,'fit':fit,'checkpoint':meta,'float32_check_replay':exact}
                del fitted,restored_fit
            row['seconds'] = time.perf_counter()-start
            write(target/'result.json',row)
            print(f"LOCAL seed{seed} block{index} {case}: ideal={ideal['check_mse']:.6g} physical={physical['check_mse']:.6g} ({row['seconds']:.1f}s)",flush=True)
            del selected,block,restored,ideal_moments
            report(config,output,'running',config.get('synthetic',False))
        del packs,moments
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def full_models(config,baseline,source,tensors,targets,mean,std,seed,output,state,stop):
    device = mean.device
    for case in config['full_cases']:
        stop()
        folder = output/f'seed_{seed}'/'models'/case
        if (folder/'result.json').exists():
            row = read(folder/'result.json')
            verified(folder/'model.pt',row['checkpoint'])
            verified(folder/'logits.pt',row['logits'])
            continue
        model = deepcopy(source)
        for index in config['blocks']:
            local = output/f'seed_{seed}'/f'block_{index}'/case
            row = read(local/'result.json')
            model.blocks[index] = restore_block(source.blocks[index],local/'block.pt',row['checkpoint'],device)
        state(stage='before_repair',seed=seed,case=case)
        before = metrics(model,tensors,targets,mean,std,config['repair']['batch_size'])
        logits = {'before':{s:torch.from_numpy(evaluate(model,tensors[s],targets[s],mean,std,config['repair']['batch_size'])[1]) for s in before}}
        probe = deepcopy(model).train()
        audit = first_update_audit(probe,prepare_batch(tensors['train'][:8],mean,std),targets['train'][:8],torch.optim.SGD(probe.parameters(),lr=.001))
        del probe
        def on_epoch(row,counters):
            state(stage='repair',seed=seed,case=case,epoch=row['epoch'],validation=row['validation'],counts=counters)
            print(f"REPAIR seed{seed} {case}: {row['epoch']}/{config['repair']['epochs']} val={100*row['validation']['accuracy']:.2f}% ({row['epoch_seconds']:.1f}s)",flush=True)
        model,history,counters,done = repair(model,tensors,targets,mean,std,config['repair'],seed,folder/'last.pt',output/'STOP',on_epoch)
        if not done:
            raise Paused()
        after = metrics(model,tensors,targets,mean,std,config['repair']['batch_size'])
        logits['after'] = {s:torch.from_numpy(evaluate(model,tensors[s],targets[s],mean,std,config['repair']['batch_size'])[1]) for s in after}
        logits['labels'] = {s:targets[s].cpu() for s in after}
        digest = save_model(model,folder/'model.pt')
        replica = load_model(folder/'model.pt',baseline,device)
        assert_replay(after,metrics(replica,tensors,targets,mean,std,config['repair']['batch_size']))
        atomic_save(logits,folder/'logits.pt')
        count = sum(p.numel() for p in model.parameters())
        if not config.get('synthetic') and count != 232266:
            raise AssertionError('Compact model parameter count changed')
        row = {'seed':seed,'case':case,'before':before,'after':after,'history':history,'optimizer_counts':counters,
               'real_trainable_parameters':count,'checkpoint_bytes':(folder/'model.pt').stat().st_size,
               'checkpoint':{'sha256':digest},'logits':{'sha256':file_sha256(folder/'logits.pt')},
               'discarded_copy_update_audit':audit,'metric_reload_replay_passed':True,'test_evaluated':False}
        write(folder/'result.json',row)
        report(config,output,'running',config.get('synthetic',False))
        del model,replica,logits
        gc.collect()


def execute(config,baseline,sources,tensors,targets,mean,std,output):
    output.mkdir(parents=True,exist_ok=True)
    def stop():
        if (output/'STOP').exists():
            raise Paused()
    def state(**fields):
        write(output/'state.json',{'status':'running','pid':os.getpid(),**fields,'updated_utc':now()})
    try:
        # Finish all local mechanism checks before supervised network repair.
        for seed in config['seeds']:
            source = sources(seed)
            local_studies(config,baseline,source,tensors,targets,mean,std,seed,output,state,stop)
            del source
        for seed in config['seeds']:
            source = sources(seed)
            full_models(config,baseline,source,tensors,targets,mean,std,seed,output,state,stop)
            del source
        expected_local = len(config['seeds'])*len(config['blocks'])*(3+2*config['random_repeats'])
        result = report(config,output,'completed',config.get('synthetic',False))
        if len(result['local_results']) != expected_local or len(result['results']) != len(config['seeds'])*len(config['full_cases']):
            raise AssertionError('Study incomplete')
        checked = 0
        for p in output.glob('seed_*/block_*/*/result.json'):
            row = read(p)
            verified(p.parent/'block.pt',row['checkpoint']); checked += 1
            if 'datafit' in row:
                verified(p.parent/'datafit_block.pt',row['datafit']['checkpoint']); checked += 1
        for p in output.glob('seed_*/models/*/result.json'):
            row = read(p)
            for name,key in (('model.pt','checkpoint'),('logits.pt','logits')):
                verified(p.parent/name,row[key]); checked += 1
        write(output/'completion_audit.json',{'status':'passed','artifact_hashes_checked':checked,'updated_utc':now()})
        write(output/'state.json',{'status':'completed','local_cases':expected_local,'models':len(result['results']),'updated_utc':now()})
        return result
    except Paused:
        report(config,output,'paused',config.get('synthetic',False))
        write(output/'state.json',{'status':'paused','updated_utc':now()})
        return None
    except BaseException as error:
        report(config,output,'failed',config.get('synthetic',False))
        write(output/'state.json',{'status':'failed','error':f'{type(error).__name__}: {error}','updated_utc':now()})
        raise


def smoke(config):
    tiny = deepcopy(config)
    tiny.update(seeds=[19],random_repeats=1,device='cpu',synthetic=True,collection_batch_size=2)
    tiny['repair'].update(epochs=1,batch_size=4,mixed_precision=False)
    tiny['producer_fit'].update(steps=3,batch_size=2)
    baseline = read(ROOT/config['baseline_config'])
    baseline.update(quaternion_widths=[8,8,8],classes=2)
    seed_everything(19)
    source = make_model(baseline).eval()
    sizes = {'train':8,'validation':4,'fit':4,'check_calibration':4}
    tensors = {s:torch.randint(256,(n,3,8,8),dtype=torch.uint8) for s,n in sizes.items()}
    targets = {s:torch.arange(n)%2 for s,n in sizes.items()}
    output = ROOT/'.cache'/('grouping_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    args = (tiny,baseline,lambda seed:deepcopy(source),tensors,targets,torch.zeros(1,3,1,1),torch.ones(1,3,1,1),output)
    result = execute(*args)
    replay = execute(*args)
    if result['results'] != replay['results'] or result['local_results'] != replay['local_results']:
        raise AssertionError('Completed-study resume changed results')
    return {'status':'passed','output':output.relative_to(ROOT).as_posix(),'local_cases':10,'models':5}


def prepare(config):
    output = ROOT/config['output']
    record = {'started_utc':now(),'scientific_results':False}
    try:
        tests = subprocess.run([sys.executable,'-B','-m','pytest','-q','tests_torch/test_grouping_mechanism.py',
                                'tests_torch/test_resnet_reassembly.py','tests_torch/test_component_reassembly.py'],cwd=ROOT,capture_output=True,text=True)
        record['tests'] = {'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr}
        print(tests.stdout,flush=True)
        if tests.returncode:
            raise RuntimeError('Grouping tests failed')
        with lock('compression_curve_runner.lock'):
            record['smoke'] = smoke(config)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            baseline = read(ROOT/config['baseline_config'])
            model = make_model(baseline).cuda().eval()
            for index in config['blocks']:
                width = model.blocks[index].conv1.out_channels
                groups = case_groups(list(range(width//2)),width,19,index,config)['free_100_0']
                model.blocks[index],_ = compile_block(model.blocks[index],groups)
            x = torch.randn(8,3,32,32,device='cuda')
            y = torch.arange(8,device='cuda')%baseline['classes']
            record['gpu_fp32_audit'] = first_update_audit(model,x,y,torch.optim.SGD(model.parameters(),lr=.001))
            del model,x,y
            # Exercise the new producer-only optimizer on GPU before scientific data.
            from src.models.cifar_resnet import BasicBlock
            original = BasicBlock('quaternion',2,8).cuda().eval()
            x = torch.randn(8,8,8,8)
            with torch.no_grad():
                h = torch.relu(original.bn1(original.conv1(x.cuda()))).cpu()
            groups = case_groups([0,1,2,3],8,19,0,config)['free_100_0']
            block,_ = compile_block(original,groups)
            local = {'inputs':x,'selected_features':h[:,packing_indices(groups,8)]}
            _,record['gpu_producer_audit'] = fit_producer(block,local,{**config['producer_fit'],'steps':3,'batch_size':4},19,'cuda')
            del original,block,x,h,local
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
    print('READY: grouping study registered before scientific fitting',flush=True)


def run(config):
    with lock('compression_curve_runner.lock'):
        definition = register(config)
        output = ROOT/config['output']
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Readiness does not cover these sources')
        device = torch.device(config['device'])
        baseline = read(ROOT/config['baseline_config'])
        tensors,targets,mean,std = load_data(baseline,device)
        replays = {}
        def sources(seed):
            seed_everything(seed)
            model = make_model(baseline).to(device).eval()
            folder = parent_folder(seed)
            model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True)['model'])
            actual = metrics(model,tensors,targets,mean,std,config['repair']['batch_size'])
            assert_replay({'validation':read(folder/'metrics.json')['selected_validation']},{'validation':actual['validation']})
            replays[str(seed)] = actual
            write(output/'parent_validation_replays.json',replays)
            return model
        result = execute(config,baseline,sources,tensors,targets,mean,std,output)
        if result is not None:
            register(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    study = read(ROOT/CONFIG)
    torch.set_num_threads(study['cpu_threads'])
    run(study) if args.run else prepare(study)
