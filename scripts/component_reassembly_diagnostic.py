"""Exhaustive tiny CPU check of physically compiled quaternion reassembly.

--prepare runs technical tests and freezes the protocol; --run evaluates the
two prespecified synthetic cases. No dataset training or novelty claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from src.compression.component_reassembly import (TinyQuaternionPair, compiled_pair,
                                                parameter_counts, refit_consumer)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/component_reassembly_diagnostic.json'
SOURCES = [CONFIG,'scripts/component_reassembly_diagnostic.py','src/compression/component_reassembly.py',
           'tests_torch/test_component_reassembly.py','src/models/quaternion_conv_torch.py','src/cnn_training.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def register(config):
    if (config['input_quaternion_units'],config['source_quaternion_units'],config['retained_quaternion_units'],config['output_quaternion_units']) != (1,2,1,1):
        raise ValueError('Only the bounded 8-to-4 scalar diagnostic is implemented')
    output = ROOT/config['output']
    definition = {'config':config,'source_hashes':{p:sha(ROOT/p) for p in SOURCES},
                  'environment':{'python':platform.python_version(),'torch':torch.__version__},
                  'synthetic_only':True,'test_evaluated':False}
    path = output/'protocol.json'
    if path.exists():
        if read(path) != definition:
            raise ValueError('Frozen diagnostic changed; register a new version')
    else:
        write(path,definition)
    return definition


def make_case(config,case):
    generator = torch.Generator().manual_seed(config['seed'])
    teacher = TinyQuaternionPair().double().eval()
    a = torch.randn(4,generator=generator,dtype=torch.float64)
    a /= a.norm()
    noise = torch.randn(4,generator=generator,dtype=torch.float64)
    b = a + case['producer_noise']*noise if case['producer_relation'] == 'nearby' else noise/noise.norm()
    consumer = torch.randn(4,generator=generator,dtype=torch.float64)
    consumer /= consumer.norm()
    with torch.no_grad():
        teacher.producer.kernel[:,0,0,0,0].copy_(a)
        teacher.producer.kernel[:,1,0,0,0].copy_(b)
        teacher.consumer.kernel[:,0,:,0,0].copy_(consumer[:,None].expand(4,2))
        positive,negative = config['bn_positive_bias'],config['bn_negative_bias']
        teacher.bn.bias.copy_(torch.tensor([positive,negative,negative,positive,positive,negative,negative,positive],dtype=torch.float64))
    def draw(count,offset):
        return torch.randn(count,4,1,1,generator=torch.Generator().manual_seed(config['seed']+offset),dtype=torch.float64)
    return teacher,draw(config['fit_examples'],100),draw(config['check_examples'],200)


@torch.no_grad()
def candidate(teacher,quartet,x,targets,teacher_features,alpha):
    student,metrics = compiled_pair(teacher,[quartet])
    ideal = refit_consumer(student,teacher_features[:,list(quartet)],targets,alpha)
    actual_features = student.features(x)
    actual = refit_consumer(student,actual_features,targets,alpha)
    row = {'quartet':list(quartet),**metrics,'ideal_fit_mse':ideal,'realizable_fit_mse':actual,
           'feature_fit_mse':float((actual_features-teacher_features[:,list(quartet)]).square().mean()),
           'mixed_original_units':len({index%2 for index in quartet})>1}
    return student,row


def report(config,rows,status):
    output = ROOT/config['output']
    write(output/'summary.json',{'status':status,'config':config,'results':rows,'synthetic_only':True,
                                'test_evaluated':False,'automatic_followup':False,'updated_utc':now()})
    lines = ['# Component reassembly: synthetic realizability diagnostic','',f'Status: {status}.','',
             '## Hypothesis','',config['hypothesis'],'','## Measured results','',
             '| Synthetic case | Selection | Quartet indices | Fit MSE | Held-out MSE | Producer projection SSE | Stored reals |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['method']} | {r['quartet']} | {r['realizable_fit_mse']:.6g} | {r['check_mse']:.6g} | {r['producer_kernel_squared_error']:.6g} | {r['counts']['real_parameters']} |")
    lines += ['', '## Mathematical construction','',
              'For target scalar producer rows b_a assigned roles a=r/i/j/k, let T_a',
              'be the signed permutation taking quaternion coefficients to output row a.',
              'Pull them back as c_a=T_a^T b_a. The optimal shared coefficients are',
              'their mean. Frobenius projection error is sum_a||c_a-mean(c)||^2,',
              'equivalently one quarter of sum_{a<b}||c_a-c_b||^2. Original intact',
              'quartets have zero error. Arbitrary regrouped quartets generally do not.',
              'This identity is elementary projection algebra, not a novelty claim.','',
              'The consumer is fitted on actual compact-producer BN/ReLU outputs.',
              'Full-teacher selected activations are an optimistic diagnostic only.',
              'The final forward contains one smaller producer, its BN/ReLU and one',
              'smaller consumer; no full teacher, gather, router or dense adapter.','',
              '## Interpretation and limits','',config['scope'],'',config['construction'],'',
              config['producer_fit'],'',config['selection'],'',config['interpretation'],'',
              'Teacher:32real parameters; all compiled students:16, including BN affine.',
              'No labels, classifier accuracy, fine-tuning or CIFAR inference were used.',
              'These deliberately constructed examples are not evidence of real-data gains.',
              'Prior art: '+', '.join(config['prior_art'])+' (references.bib).','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def prepare(config):
    checked = subprocess.run([sys.executable,'-B','-m','pytest','-q','tests_torch/test_component_reassembly.py'],
                             cwd=ROOT,capture_output=True,text=True)
    print(checked.stdout,flush=True)
    if checked.returncode:
        raise RuntimeError(checked.stdout+checked.stderr)
    definition = register(config)
    output = ROOT/config['output']
    write(output/'readiness.json',{'status':'passed','source_hashes':definition['source_hashes'],
                                 'tests':checked.stdout.strip(),'synthetic_only':True,'updated_utc':now()})
    write(output/'state.json',{'status':'ready_not_started','updated_utc':now()})
    print('READY: technical protocol frozen before case enumeration',flush=True)


@torch.no_grad()
def run(config):
    output = ROOT/config['output']
    definition = register(config)
    ready = read(output/'readiness.json')
    if ready['status'] != 'passed' or ready['source_hashes'] != definition['source_hashes']:
        raise ValueError('Readiness mismatch')
    rows = []
    try:
        for case in config['cases']:
            teacher,x,check = make_case(config,case)
            targets,teacher_features = teacher(x),teacher.features(x)
            folder = output/case['name']
            folder.mkdir(parents=True,exist_ok=True)
            torch.save({'teacher':teacher.state_dict(),'fit_inputs':x,'check_inputs':check},folder/'synthetic_inputs.pt')
            records = []
            for number,quartet in enumerate(itertools.permutations(range(8),4),1):
                if (output/'STOP').exists():
                    report(config,rows,'paused')
                    write(output/'state.json',{'status':'paused','case':case['name'],'updated_utc':now()})
                    return
                _,record = candidate(teacher,quartet,x,targets,teacher_features,config['ridge_alpha'])
                records.append(record)
                if number%240 == 0:
                    write(output/'state.json',{'status':'running','pid':os.getpid(),'case':case['name'],
                                              'evaluated_candidates':number,'updated_utc':now()})
                    print(f"{case['name']}: {number}/1680 quartets",flush=True)
            if len(records) != 1680:
                raise AssertionError('Incomplete enumeration')
            write(folder/'candidates.json',records)
            by_quartet = {tuple(r['quartet']):r for r in records}
            whole = [by_quartet[(unit,unit+2,unit+4,unit+6)] for unit in range(2)]
            energy = teacher_features.square().mean((0,2,3))
            selected = sorted(sorted(range(8),key=lambda i:(-float(energy[i]),i))[:4])
            selections = {
                'best_whole_unit':min(whole,key=lambda r:(r['realizable_fit_mse'],r['quartet'])),
                'scalar_energy':by_quartet[tuple(selected)],
                'consumer_only_selection':min(records,key=lambda r:(r['ideal_fit_mse'],r['quartet'])),
                'realizable_selection':min(records,key=lambda r:(r['realizable_fit_mse'],r['quartet'])),
            }
            for method,selected in selections.items():
                student,record = candidate(teacher,tuple(selected['quartet']),x,targets,teacher_features,config['ridge_alpha'])
                if record['realizable_fit_mse'] != selected['realizable_fit_mse']:
                    raise AssertionError('Selected fit did not reproduce')
                path = folder/f'{method}.pt'
                torch.save({'model':student.state_dict(),'quartet':selected['quartet']},path)
                replica = TinyQuaternionPair(hidden=1).double().eval()
                replica.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
                torch.testing.assert_close(replica(check),student(check),atol=0,rtol=0)
                count = parameter_counts(student)
                if count['real_parameters'] != 16:
                    raise AssertionError('Full producer or extra module remains')
                ideal_student,_ = compiled_pair(teacher,[tuple(selected['quartet'])])
                refit_consumer(ideal_student,teacher_features[:,selected['quartet']],targets,config['ridge_alpha'])
                ideal_check = ideal_student.consumer(teacher.features(check)[:,selected['quartet']])
                result = {'case':case['name'],'method':method,**record,'counts':count,
                          'check_mse':float((student(check)-teacher(check)).square().mean()),
                          'ideal_check_mse':float((ideal_check-teacher(check)).square().mean()),
                          'checkpoint_sha256':sha(path),'checkpoint_bytes':path.stat().st_size,
                          'reload_exact':True,'synthetic_only':True}
                rows.append(result)
                write(folder/f'{method}.json',result)
                print(f"{case['name']} {method}: fit={result['realizable_fit_mse']:.6g} check={result['check_mse']:.6g} quartet={record['quartet']}",flush=True)
            report(config,rows,'running')
        register(config)
        report(config,rows,'completed')
        write(output/'state.json',{'status':'completed','synthetic_only':True,'cases':len(config['cases']),
                                  'candidate_fits':3360,'compiled_models':len(rows),'updated_utc':now()})
    except BaseException as error:
        write(output/'state.json',{'status':'failed','error':f'{type(error).__name__}: {error}','updated_utc':now()})
        report(config,rows,'failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    config = read(ROOT/CONFIG)
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    if args.run:
        run(config)
    else:
        prepare(config)
