"""Measure complete prompt budgets without loading model weights."""
from __future__ import annotations

import argparse
from pathlib import Path

from clumsification_code.data.io import write_json_atomic, write_jsonl_atomic
from clumsification_code.data.repository import DatasetRepository
from clumsification_code.perturbations.generation import PerturbationGenerationService
from clumsification_code.perturbations.generation_config import prepare_request
from clumsification_code.perturbations.length_planning import load_tokenizer, plan_token_budgets, budget_report
from clumsification_code.perturbations.batch_store import fingerprint
from dataclasses import asdict
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True)
    p.add_argument("--dataset-root",type=Path,default=Path("data/custom_datasets"))
    p.add_argument("--method",choices=["llm_single","llm_sampled"],required=True)
    p.add_argument("--assignment-file",required=True)
    p.add_argument("--model-path",default="Qwen/Qwen3.8-27B")
    p.add_argument("--tokenizer",default=None)
    p.add_argument("--revision",default=None)
    p.add_argument("--max-model-len",type=int,default=32768)
    p.add_argument("--source-partitions",nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    config={"model":a.model_path,"assignment_file":a.assignment_file,"max_model_len":a.max_model_len}
    if a.tokenizer: config["tokenizer"]=a.tokenizer
    if a.revision: config["revision"]=a.revision
    repo=DatasetRepository.from_root(a.dataset_root,a.dataset)
    request=prepare_request(dataset_name=a.dataset,source_layer=0,source_method=None,source_run_id=None,
                            method=a.method,run_id="profile",target_layer=None,config=config,
                            source_partitions=tuple(a.source_partitions or ()),limit=None)
    items=PerturbationGenerationService(repo).load_source_items(source_layer=0,source_method=None,source_run_id=None,
                                                              source_partitions=tuple(a.source_partitions or ()) or None)
    adapter=request.spec.create(request.method_config)
    prompts=adapter.build_prompts([i.metadata|{"base_text_id":i.base_text_id,"candidate_id":i.candidate_id,"text":i.text} for i in items])
    tokenizer=load_tokenizer(a.model_path,config)
    budgets,bounds=plan_token_budgets(prompts,[i.text.replace("\n"," ") for i in items],tokenizer,request.method_config)
    report=budget_report(budgets,bounds)
    report["fingerprint"], report["provenance"]=fingerprint(request,items)
    report["tokenizer_class"]=type(tokenizer).__name__
    report["recommended_config"]={"source_buckets":list(bounds),"max_model_len":a.max_model_len}
    write_json_atomic(a.output,report,overwrite=True)
    write_jsonl_atomic(a.output.with_suffix(".lengths.jsonl"),
                      [{"parent_candidate_id":i.candidate_id,**asdict(b)} for i,b in zip(items,budgets)],overwrite=True)
    print(json.dumps({k:v for k,v in report.items() if k!="provenance"},indent=2))


if __name__=="__main__": main()
