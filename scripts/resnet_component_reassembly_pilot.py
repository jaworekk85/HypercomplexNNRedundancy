"""Registered two-block Q ResNet component-reassembly pilot; prepare then run."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.run_activation_queue import lock
from src.cnn_training import evaluate, first_update_audit, prepare_batch, seed_everything
from src.compression.resnet_core import choose, compact, refit
from src.compression.resnet_reassembly import (compile_block, energy_groups, final_consumer_fit,
    local_mse, make_catalogue, original_quartets, positions_for, select_groups, selection_statistics)
from src.data.cifar_cached import file_sha256, load_cached_cifar
from src.models.cifar_resnet import CifarResNet20
from src.trainable_map_repair import atomic_save, repair

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/resnet_component_reassembly_pilot.json'
NEW = [CONFIG,'scripts/resnet_component_reassembly_pilot.py','src/compression/resnet_reassembly.py',
       'tests_torch/test_resnet_reassembly.py','src/compression/component_reassembly.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def make_model(baseline,hidden=None):
    return CifarResNet20('quaternion',baseline['real_widths'],baseline['quaternion_widths'],baseline['classes'],hidden)


def save_model(model,path):
    atomic_save({'model':{n:v.detach().cpu() for n,v in model.state_dict().items()},
                 'hidden_widths':[b.conv1.out_channels for b in model.blocks]},path)
    return file_sha256(path)


def load_model(path,baseline,device):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    model = make_model(baseline,saved['hidden_widths']).to(device)
    model.load_state_dict(saved['model'])
    return model.eval()


def register(study):
    parent = read(ROOT/study['parent_protocol'])
    for name,digest in {**parent['source_hashes'],**parent['input_hashes']}.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError(f'Frozen predecessor changed: {name}')
    metrics = read(ROOT/study['baseline_metrics'])
    if file_sha256(ROOT/study['baseline_checkpoint']) != metrics['checkpoint_sha256']:
        raise ValueError('Q parent checkpoint changed')
    baseline = read(ROOT/study['baseline_config'])
    paths = [study[k] for k in ('parent_protocol','baseline_config','baseline_checkpoint','baseline_metrics')]
    definition = {'config':study,
        'source_hashes':{**parent['source_hashes'],**{p:file_sha256(ROOT/p) for p in NEW}},
        'input_hashes':{**parent['input_hashes'],**{p:file_sha256(ROOT/p) for p in paths}},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda},
        'test_evaluated':False}
    output = ROOT/study['output']
    path = output/'protocol.json'
    if path.exists():
        if read(path) != definition:
            raise ValueError('Registered reassembly pilot changed')
    else:
        write(path,definition)
    text = ['# Two-block component-reassembly protocol','',study['hypothesis'],'',
            '## Registered plan','',study['scope'],'',study['controls'],'',study['producer'],'',
            study['catalogue'],'',study['selection_data'],'',study['joint_fit'],'',study['search'],'',
            study['final_fit'],'',study['evaluation'],'','## Continuation decision','',study['decision'],'',
            '## Reproducibility','',study['resume'],'',
            'Exact numerical settings, hashes, versions and citation keys are in protocol.json.','']
    (output/'PROTOCOL.md').write_text('\n'.join(text),encoding='utf-8')
    return definition,baseline


def load_data(baseline,device):
    (images,labels),unused_test = load_cached_cifar(ROOT/baseline['archive'])
    del unused_test
    with np.load(ROOT/baseline['split_source']) as split:
        ids = {'train':split['train'],'validation':split['validation'],'fit':split['calibration'][::2],
               'check_calibration':split['calibration'][1::2]}
    sets = [set(v.tolist()) for v in ids.values()]
    if [len(ids[k]) for k in ids] != [48000,1000,500,500] or sum(map(len,sets)) != 50000 or sorted(set.union(*sets)) != list(range(50000)):
        raise ValueError('Invalid train/fit/check/validation split')
    stats = read(ROOT/baseline['data_statistics'])['statistics']
    mean,std = [torch.tensor(stats[k],device=device).reshape(1,3,1,1) for k in ('mean','std')]
    tensors = {s:torch.as_tensor(images[v],device=device) for s,v in ids.items()}
    targets = {s:torch.as_tensor(labels[v],device=device) for s,v in ids.items()}
    return tensors,targets,mean,std


class Paused(Exception):
    pass


@torch.no_grad()
def collect_source(source,index,images,mean,std,batch_size,check_stop):
    pack = {'inputs':[],'features':[],'targets':[]}
    block = source.blocks[index]
    handles = [block.conv1.register_forward_pre_hook(lambda layer,args:pack['inputs'].append(args[0].detach().cpu())),
               block.conv2.register_forward_hook(lambda layer,args,out:(pack['features'].append(args[0].detach().cpu()),pack['targets'].append(out.detach().cpu())) and None)]
    try:
        for batch in images.split(batch_size):
            check_stop()
            source(prepare_batch(batch,mean,std))
    finally:
        for handle in handles:
            handle.remove()
    return {k:torch.cat(v) for k,v in pack.items()}


def checked_pack(path,meta,build):
    if path.exists():
        if not meta.exists() or file_sha256(path) != read(meta)['sha256']:
            raise ValueError(f'Incomplete/changed cache: {path}')
        return torch.load(path,map_location='cpu',weights_only=True)
    value = build()
    atomic_save(value,path)
    write(meta,{'sha256':file_sha256(path)})
    return value


def metrics(model,tensors,targets,mean,std,batch):
    return {s:evaluate(model,tensors[s],targets[s],mean,std,batch)[0] for s in ('validation','check_calibration')}


def assert_replay(expected,actual):
    for split in expected:
        if expected[split]['accuracy'] != actual[split]['accuracy'] or abs(expected[split]['cross_entropy']-actual[split]['cross_entropy']) > 1e-6:
            raise AssertionError(f'Checkpoint metric replay failed: {split}')


def decision(study,rows):
    if len(rows) != len(study['methods']):
        return {'status':'pending'}
    measured = {r['method']:r['after']['validation'] for r in rows}
    controls = [measured[m] for m in ('legacy_direct','whole_joint','scalar_energy')]
    target = measured['free_roles']
    accuracy_gain = target['accuracy']-max(r['accuracy'] for r in controls)
    ce_gain = min(r['cross_entropy'] for r in controls)-target['cross_entropy']
    return {'status':'evaluated','continue_to_replication':accuracy_gain >= study['continuation_accuracy_gain'] and ce_gain >= 0,
            'accuracy_gain_vs_best_control':accuracy_gain,'ce_gain_vs_best_control':ce_gain,
            'primary_free_minus_whole_accuracy':target['accuracy']-measured['whole_joint']['accuracy'],
            'primary_free_minus_whole_ce':target['cross_entropy']-measured['whole_joint']['cross_entropy'],
            'statistical_significance_claimed':False,'automatic_followup':False}


def report(study,output,local,rows,status,synthetic=False):
    outcome = decision(study,rows)
    write(output/'summary.json',{'status':status,'config':study,'local_results':local,'results':rows,
          'decision':outcome,'scientific_results':not synthetic,'test_evaluated':False,'updated_utc':now()})
    lines = ['# Q ResNet component-reassembly pilot','',f'Status: {status}.','',
             '## Hypothesis','',study['hypothesis'],'','## Measured results','',
             '| Method | Actual real trainables | Validation before (%) | Validation epoch3 (%) | CE after |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['real_trainable_parameters']} | {100*r['before']['validation']['accuracy']:.2f} | {100*r['after']['validation']['accuracy']:.2f} | {r['after']['validation']['cross_entropy']:.6g} |")
    lines += ['', '| Block | Method | Mixed quartets | Changed roles | Fit MSE | Check MSE |',
              '|---|---|---:|---:|---:|---:|']
    for r in local:
        lines.append(f"| {r['block']} | {r['method']} | {r['mixed_quartets']} | {r['changed_roles']} | {r['fit_mse']:.6g} | {r['check_mse']:.6g} |")
    lines += ['', '## Interpretation and limits','',study['scope'],'',study['controls'],'',study['evaluation'],'',
              study['decision'],'','Gate outcome: `'+json.dumps(outcome)+'`.','',
              'Detailed mask/group assignments, selection traces, times and local metrics are in per-block records.',
              'No local-error ordering, novelty, equivalence or general classification gain is assumed.',
              'Prior art: '+', '.join(study['prior_art'])+' (references.bib).','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def execute(study,baseline,source,tensors,targets,mean,std,output,synthetic=False):
    output.mkdir(parents=True,exist_ok=True)
    local,rows = [],[]
    device = mean.device
    def state(**fields):
        write(output/'state.json',{'status':'running','pid':os.getpid(),**fields,'updated_utc':now()})
    def check_stop():
        if (output/'STOP').exists():
            raise Paused()
    try:
        for index in study['blocks']:
            check_stop()
            folder = output/f'block_{index}'
            folder.mkdir(parents=True,exist_ok=True)
            if all((folder/m/'result.json').exists() for m in study['methods']):
                for method in study['methods']:
                    r = read(folder/method/'result.json')
                    if file_sha256(folder/method/'block.pt') != r['checkpoint_sha256']:
                        raise ValueError('Completed block changed')
                    local.append(r)
                continue
            packs = {}
            for split in ('fit','check_calibration'):
                state(stage='source_activations',block=index,split=split)
                packs[split] = checked_pack(folder/f'{split}.pt',folder/f'{split}.json',
                    lambda split=split:collect_source(source,index,tensors[split],mean,std,study['collection_batch_size'],check_stop))
            original = source.blocks[index]
            count = max(1,round(original.conv1.out_channels*study['retained_fraction']))
            bank,meta = make_catalogue(original,packs['fit']['features'],count,study)
            catalogue = {'quartets':[list(q) for q in bank],**meta}
            if (folder/'catalogue.json').exists() and read(folder/'catalogue.json') != catalogue:
                raise ValueError('Candidate catalogue changed')
            write(folder/'catalogue.json',catalogue)
            height,width = packs['fit']['targets'].shape[-2:]
            positions = positions_for(len(packs['fit']['inputs']),height*width,study['selection_positions_per_image'],
                                      study['seed']+study['position_seed_offset']+index)
            state(stage='candidate_bank',block=index,candidates=len(bank))
            def build_stats():
                value = selection_statistics(original,bank,packs['fit'],positions,study,device,check_stop)
                return {k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in value.items()}
            stored_stats = checked_pack(folder/'statistics.pt',folder/'statistics.json',build_stats)
            stats = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in stored_stats.items()}
            del stored_stats
            for method in study['methods']:
                check_stop()
                target = folder/method
                target.mkdir(parents=True,exist_ok=True)
                if (target/'result.json').exists():
                    row = read(target/'result.json')
                    if file_sha256(target/'block.pt') != row['checkpoint_sha256']:
                        raise ValueError('Completed local model changed')
                else:
                    started = time.perf_counter()
                    state(stage='selection',block=index,method=method)
                    if method == 'legacy_direct':
                        victims,maps = choose(original,packs['fit']['features'],original.conv1.out_channels-count,study['ridge_alpha'])
                        folded,kept = compact(source,index,victims,maps)
                        rebuilt,fit_record = refit(source,folded,index,kept,
                            (packs['fit']['features'],packs['fit']['targets']),
                            (packs['check_calibration']['features'],packs['check_calibration']['targets']),study['ridge_alpha'])
                        block = rebuilt.blocks[index]
                        groups = [original_quartets(original.conv1.out_channels)[i] for i in kept]
                        _,description = compile_block(original,groups)
                        selection = {'victims':victims,'estimator':'frozen mapfold-centered legacy direct'}
                        del folded,rebuilt
                    else:
                        if method == 'scalar_energy':
                            groups = energy_groups(packs['fit']['features'],count,False)
                            selection = {'estimator':'scalar second moments; sorted component-blocked packing'}
                        else:
                            def progress(**kw):
                                state(stage='selection',block=index,method=method,**kw)
                                print(f"SELECT block{index} {method}: {kw['phase']} {kw['selected']}/{kw['target']}",flush=True)
                            groups,selection = select_groups(stats,bank,original.conv1.out_channels,count,method,study,progress,check_stop)
                        block,description = compile_block(original,groups)
                        full_penalty = stats['penalty']*(height*width/positions.shape[1])
                        state(stage='full_consumer_fit',block=index,method=method)
                        fit_record = final_consumer_fit(block,packs['fit'],full_penalty,device,study['collection_batch_size'],check_stop)
                    block = block.to(device).eval()
                    fit_mse = local_mse(block,packs['fit'],device,study['collection_batch_size'])
                    check_mse = local_mse(block,packs['check_calibration'],device,study['collection_batch_size'])
                    payload = {'model':{n:v.detach().cpu() for n,v in block.state_dict().items()},'quartets':[list(q) for q in groups]}
                    atomic_save(payload,target/'block.pt')
                    replica,_ = compile_block(original,groups)
                    replica.load_state_dict(torch.load(target/'block.pt',map_location=device,weights_only=True)['model'])
                    replay = local_mse(replica,packs['check_calibration'],device,study['collection_batch_size'])
                    if abs(replay-check_mse) > 1e-10*max(1.,check_mse):
                        raise AssertionError('Local artifact replay failed')
                    row = {'block':index,'method':method,**description,'selection':selection,'consumer_fit':fit_record,
                           'fit_mse':fit_mse,'check_mse':check_mse,'current_attempt_seconds':time.perf_counter()-started,
                           'checkpoint_sha256':file_sha256(target/'block.pt'),'local_reload_replay_mse':replay}
                    write(target/'result.json',row)
                    print(f"LOCAL block{index} {method}: checkMSE={check_mse:.6g} mixed={description['mixed_quartets']} roles={description['changed_roles']}",flush=True)
                local.append(row)
                report(study,output,local,rows,'running',synthetic)
            del packs,stats
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        for method in study['methods']:
            check_stop()
            folder = output/method
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():
                row = read(folder/'result.json')
                if file_sha256(folder/'model.pt') != row['checkpoint_sha256']:
                    raise ValueError('Completed full model changed')
            else:
                model = deepcopy(source)
                for index in study['blocks']:
                    path = output/f'block_{index}'/method/'block.pt'
                    saved = torch.load(path,map_location=device,weights_only=True)
                    groups = [tuple(q) for q in saved['quartets']]
                    replacement,_ = compile_block(source.blocks[index],groups)
                    replacement.load_state_dict(saved['model'])
                    model.blocks[index] = replacement
                before = metrics(model,tensors,targets,mean,std,study['repair']['batch_size'])
                probe = deepcopy(model).train()
                audit = first_update_audit(probe,prepare_batch(tensors['train'][:8],mean,std),targets['train'][:8],
                                           torch.optim.SGD(probe.parameters(),lr=.001))
                del probe
                def on_epoch(value,counters):
                    state(stage='repair',method=method,epoch=value['epoch'],counts=counters,validation=value['validation'])
                    print(f"REPAIR {method}: {value['epoch']}/{study['repair']['epochs']} val={100*value['validation']['accuracy']:.2f}% time={value['epoch_seconds']:.1f}s",flush=True)
                state(stage='repair',method=method,epoch=0)
                model,history,counters,done = repair(model,tensors,targets,mean,std,study['repair'],study['seed'],folder/'last.pt',output/'STOP',on_epoch)
                if not done:
                    raise Paused()
                after = metrics(model,tensors,targets,mean,std,study['repair']['batch_size'])
                digest = save_model(model,folder/'model.pt')
                replica = load_model(folder/'model.pt',baseline,device)
                replay = metrics(replica,tensors,targets,mean,std,study['repair']['batch_size'])
                assert_replay(after,replay)
                row = {'method':method,'before':before,'after':after,'history':history,'optimizer_counts':counters,
                       'first_update_discarded_copy_audit':audit,'real_trainable_parameters':sum(p.numel() for p in model.parameters()),
                       'buffer_elements':sum(b.numel() for b in model.buffers()),'checkpoint_sha256':digest,
                       'checkpoint_bytes':(folder/'model.pt').stat().st_size,'validation_reload_replay':replay,'test_evaluated':False}
                write(folder/'result.json',row)
                del model,replica
            rows.append(row)
            report(study,output,local,rows,'running',synthetic)
        if len({r['real_trainable_parameters'] for r in rows}) != 1:
            raise AssertionError('Final architecture counts differ')
        if not synthetic and rows[0]['real_trainable_parameters'] != 232266:
            raise AssertionError('Unexpected scientific compact count')
        report(study,output,local,rows,'completed',synthetic)
        write(output/'state.json',{'status':'completed','local_cases':len(local),'models':len(rows),'updated_utc':now()})
        return read(output/'summary.json')
    except Paused:
        report(study,output,local,rows,'paused',synthetic)
        write(output/'state.json',{'status':'paused','updated_utc':now()})
        return None
    except BaseException as error:
        write(output/'state.json',{'status':'failed','error':f'{type(error).__name__}: {error}','updated_utc':now()})
        report(study,output,local,rows,'failed',synthetic)
        raise


def smoke(study):
    tiny = deepcopy(study)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    output = ROOT/'.cache'/f'reassembly_smoke_{stamp}'
    tiny.update(device='cpu',role_catalogue_limit=12,full_catalogue_limit=20,
                bank_batch_size=2,collection_batch_size=2,selection_positions_per_image=2)
    tiny['repair'].update(epochs=1,batch_size=4,mixed_precision=False)
    baseline = read(ROOT/study['baseline_config'])
    baseline.update(quaternion_widths=[2,4,8],classes=2)
    seed_everything(19)
    source = make_model(baseline).eval()
    sizes = {'train':8,'validation':4,'fit':4,'check_calibration':4}
    tensors = {s:torch.randint(256,(n,3,8,8),dtype=torch.uint8) for s,n in sizes.items()}
    targets = {s:torch.arange(n)%2 for s,n in sizes.items()}
    result = execute(tiny,baseline,source,tensors,targets,torch.zeros(1,3,1,1),torch.ones(1,3,1,1),output,True)
    if result['status'] != 'completed' or len(result['local_results']) != 10 or len(result['results']) != 5:
        raise AssertionError('Incomplete synthetic end-to-end check')
    # Reuse completed cases through the same verified resume path.
    replay = execute(tiny,baseline,source,tensors,targets,torch.zeros(1,3,1,1),torch.ones(1,3,1,1),output,True)
    if replay['results'] != result['results']:
        raise AssertionError('Completed-case resume changed results')
    return {'status':'passed','output':output.relative_to(ROOT).as_posix(),'scientific_results':False}


def prepare(study):
    output = ROOT/study['output']
    record = {'started_utc':now(),'scientific_results':False}
    try:
        tests = subprocess.run([sys.executable,'-B','-m','pytest','-q','tests_torch/test_resnet_reassembly.py',
                                'tests_torch/test_component_reassembly.py'],cwd=ROOT,capture_output=True,text=True)
        print(tests.stdout,flush=True)
        record['tests'] = {'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr}
        if tests.returncode:
            raise RuntimeError('Reassembly tests failed')
        with lock('compression_curve_runner.lock'):
            record['synthetic_smoke'] = smoke(study)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            baseline = read(ROOT/study['baseline_config'])
            seed_everything(study['seed'])
            model = make_model(baseline).cuda().train()
            for index in study['blocks']:
                width = model.blocks[index].conv1.out_channels
                groups = [tuple(r*width+(i+r)%width for r in range(4)) for i in range(width//2)]
                model.blocks[index],_ = compile_block(model.blocks[index],groups)
            model.train()
            x = torch.randn(study['repair']['batch_size'],3,32,32,device='cuda')
            y = torch.arange(len(x),device='cuda')%baseline['classes']
            audit = first_update_audit(model,x[:8],y[:8],torch.optim.SGD(model.parameters(),lr=.001))
            optimizer = torch.optim.AdamW(model.parameters(),lr=study['repair']['learning_rate'])
            scaler = torch.amp.GradScaler('cuda')
            successful = []
            optimizer.register_step_post_hook(lambda *args:successful.append(True))
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    loss = torch.nn.functional.cross_entropy(model(x),y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite AMP preflight loss')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if not successful:
                raise RuntimeError('All AMP updates skipped')
            record['gpu_discarded_copy_audit'] = {'fp32':audit,'amp_attempted':3,'amp_successful':len(successful),
                'real_trainable_parameters':sum(p.numel() for p in model.parameters()),'device':torch.cuda.get_device_name()}
            del model,optimizer,scaler,x,y
            torch.cuda.empty_cache()
            definition,_ = register(study)
            write(output/'readiness.json',{'status':'passed','source_hashes':definition['source_hashes'],**record,'updated_utc':now()})
            write(output/'state.json',{'status':'ready_not_started','updated_utc':now()})
        record['status'] = 'passed'
    except BaseException as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        history_path = output/'development_checks.json'
        history = read(history_path) if history_path.exists() else []
        history.append(record)
        write(history_path,history)
    print('READY: protocol frozen; scientific fitting not yet started',flush=True)


def run(study):
    with lock('compression_curve_runner.lock'):
        definition,baseline = register(study)
        output = ROOT/study['output']
        ready = read(output/'readiness.json')
        if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
            raise ValueError('Readiness does not cover these sources')
        write(output/'state.json',{'status':'running','stage':'parent_replay','pid':os.getpid(),'updated_utc':now()})
        device = torch.device(study['device'])
        seed_everything(study['seed'])
        source = make_model(baseline).to(device).eval()
        source.load_state_dict(torch.load(ROOT/study['baseline_checkpoint'],map_location=device,weights_only=True)['model'])
        tensors,targets,mean,std = load_data(baseline,device)
        parent = metrics(source,tensors,targets,mean,std,study['repair']['batch_size'])
        expected = read(ROOT/study['baseline_metrics'])['selected_validation']
        assert_replay({'validation':expected},{'validation':parent['validation']})
        write(output/'parent_validation_replay.json',parent)
        result = execute(study,baseline,source,tensors,targets,mean,std,output)
        if result is not None:
            register(study)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    arguments = parser.parse_args()
    configuration = read(ROOT/CONFIG)
    torch.set_num_threads(configuration['cpu_threads'])
    if arguments.run:
        run(configuration)
    else:
        prepare(configuration)
