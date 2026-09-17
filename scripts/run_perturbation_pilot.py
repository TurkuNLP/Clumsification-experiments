"""Run both prepared pilot methods with one persistent runner; then build review HTML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from clumsification_code.perturbations.generation import PerturbationGenerationService
from clumsification_code.perturbations.vllm_runner import VLLMRunner
from clumsification_code.perturbations.parallel_runner import ParallelLLMRunner
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.io import write_json_atomic
from scripts.review_perturbation_pilot import render_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='fe-pilot-v3');p.add_argument('--dataset-root',type=Path,default=Path('data/custom_datasets'))
    p.add_argument('--run-id',default='pilot-v3');p.add_argument('--method-config',type=Path,default=Path('configs/perturbations/qwen38.json'))
    p.add_argument('--device-groups');p.add_argument('--accelerator',choices=['cuda','rocm'],default='cuda')
    p.add_argument('--fresh',action='store_true',help='Require an unused run ID for valid benchmark timing')
    p.add_argument('--tokenizer');p.add_argument('--revision');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--output-dir',type=Path,default=Path('analysis_outputs/perturbation_pilot'))
    a=p.parse_args()
    config=json.loads(a.method_config.read_text())
    repo=DatasetRepository.from_root(a.dataset_root,a.dataset)
    if a.fresh and any((repo.run_root(method,a.run_id)).exists() for method in ('llm_single','llm_sampled')):
        raise ValueError('--fresh requires a new run ID')
    config['assignment_file']=str(repo.dataset_dir/'perturbation_assignments.jsonl')
    config['device_groups']=a.device_groups
    config['accelerator']=a.accelerator
    if a.device_groups:
        config['tensor_parallel_size']=len(a.device_groups.split(';')[0].split(','))
        config['replicas']=len(a.device_groups.split(';'))
    if a.tokenizer: config['tokenizer']=a.tokenizer
    if a.revision: config['revision']=a.revision
    runner=ParallelLLMRunner(a.device_groups,accelerator=a.accelerator) if a.device_groups else VLLMRunner()
    started=time.monotonic(); entries=[]
    try:
        for method in ('llm_single','llm_sampled'):
            entry=PerturbationGenerationService(repo,llm_runner=runner).generate_layer(
                source_layer=0,source_method=None,source_run_id=None,method=method,run_id=a.run_id,
                config=config,source_partitions=(method,),retry_failed=a.retry_failed)
            entries.append(entry)
            write_json_atomic(a.output_dir/f'{a.run_id}.{method}.profile.json',runner.profile,overwrite=True)
    finally:
        runner.close()
        render_review(repo,a.output_dir/f'{a.run_id}.review.html',a.run_id)
    elapsed=time.monotonic()-started
    report={'run_id':a.run_id,'elapsed_seconds':elapsed,'output_count':sum(e.output_count for e in entries),
            'failure_count':sum(e.config['unresolved_failure_count'] for e in entries),
            'outputs_per_node_hour':sum(e.output_count for e in entries)*3600/elapsed,
            'device_groups':a.device_groups,'configuration':config,
            'note':'Rate is meaningful for a fresh run; resumed runs may include earlier successes.'}
    write_json_atomic(a.output_dir/f'{a.run_id}.timing.json',report,overwrite=True)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
