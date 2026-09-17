"""Build a standalone review page with severity triplets, edit counts, and local notes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import difflib
import html
import json
from pathlib import Path
import re

from clumsification_code.data.repository import DatasetRepository
from clumsification_code.data.io import read_jsonl, write_json_atomic


def render_review(repo, output, run_id):
    originals={r.base_text_id:r for r in repo.read_originals()}
    cases=read_jsonl(repo.dataset_dir/"pilot_cases.jsonl")
    results={}; failures={}
    for entry in repo.list_layers(run_ids=[run_id]):
        for row in repo.read_candidates(entry): results[row.base_text_id]=row
    # Committed batches can be inspected before final canonical publication.
    for method in ("llm_single","llm_sampled"):
        root=repo.run_root(method,run_id)/"1.batches"
        for path in sorted(root.glob("batch-*.json")):
            batch=json.loads(path.read_text())
            from clumsification_code.data.schemas import CandidateRecord
            for row in batch["successes"]:
                record=CandidateRecord.from_row(row); results[record.base_text_id]=record
            for row in batch["failures"]: failures[row["parent_candidate_id"]]=row["reason"]
    groups=defaultdict(list)
    for case in cases:
        groups[(case["pilot_source_id"],tuple(case["edits"]),case["pilot_group"],case["pilot_length"])].append(case)
    e=html.escape
    cards=[]
    for key,values in groups.items():
        source=originals[values[0]["case_id"]].text
        columns=[f'<div class="source"><h3>Original</h3><pre>{e(source)}</pre></div>']
        for severity in ("weak","medium","strong"):
            case=next((v for v in values if v["severity"]==severity),None)
            if case is None: continue
            identity=case["case_id"]; result=results.get(identity)
            if result:
                counts=json.dumps(result.metadata.get("reported_applied_edits", "unavailable"),ensure_ascii=False)
                details=f'Length change: {len(result.text)-len(source):+d} characters · Finish: {result.metadata.get("finish_reason","unknown")} · Reported counts: {counts}'
                tokens=re.findall(r'\s+|\S+',source); edited=re.findall(r'\s+|\S+',result.text)
                changes=[]
                for op,i,j,k,l in difflib.SequenceMatcher(None,tokens,edited,autojunk=False).get_opcodes():
                    if op=='equal': changes.append(e(''.join(edited[k:l])))
                    else:
                        if op in ('delete','replace'): changes.append('<del>'+e(''.join(tokens[i:j]))+'</del>')
                        if op in ('insert','replace'): changes.append('<ins>'+e(''.join(edited[k:l]))+'</ins>')
                body=f'<pre>{e(result.text)}</pre><details><summary>Text differences</summary><pre>{"".join(changes)}</pre></details>'
            else:
                details=failures.get(f'original__{repo.dataset_name}__base_{identity}', 'Pending')
                body='<p>No usable output yet.</p>'
            columns.append(f'<div data-severity="{severity}"><h3>{severity.title()}</h3><small>{e(details)}</small>{body}<label>Review notes<textarea data-case="{e(identity)}" placeholder="Meaning preserved? Edit realized? Severity appropriate?"></textarea></label></div>')
        title=f'{", ".join(key[1])} · {key[2]} · {key[3]} · source {key[0]}'
        cards.append(f'<section data-search="{e(title)}"><h2>{e(title)}</h2><div class="grid">{"".join(columns)}</div></section>')
    report={"cases":len(cases),"successful":len(results),"pending_or_failed":len(cases)-len(results),
            "run_id":run_id,"reported_counts_are_verified":False,
            "single_edit_severity_cells":len({(edit,c['severity']) for c in cases if c['pilot_group']=='single' for edit in c['edits']})}
    output.parent.mkdir(parents=True,exist_ok=True)
    page='''<!doctype html><html><head><meta charset="utf-8"><title>FE perturbation pilot review</title>
<style>body{font:15px system-ui;background:#f4f6f8;color:#172536;margin:24px}header{position:sticky;top:0;background:#f4f6f8;padding:12px;z-index:1}h1{margin:0}section{background:white;border-radius:10px;margin:20px 0;padding:18px}h2{font-size:16px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui;max-height:65vh;overflow:auto}textarea{display:block;width:95%;min-height:75px}small{display:block;color:#496079;overflow-wrap:anywhere}input,button{padding:9px;margin:8px}del{background:#ffe0e0}ins{background:#ddf5df;text-decoration:none}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style></head><body>
<header><h1>FE perturbation pilot</h1><p>SUMMARY · Counts are model-reported, not verified annotations.</p><input id="filter" placeholder="Filter operation, group, length or source"><button id="download">Download review notes</button></header>CARDS
<script>const prefix=PREFIX;document.querySelectorAll('textarea').forEach(t=>{t.value=localStorage.getItem(prefix+t.dataset.case)||'';t.oninput=()=>localStorage.setItem(prefix+t.dataset.case,t.value)});document.querySelector('#filter').oninput=e=>document.querySelectorAll('section').forEach(s=>s.hidden=!s.dataset.search.toLowerCase().includes(e.target.value.toLowerCase()));document.querySelector('#download').onclick=()=>{const notes={};document.querySelectorAll('textarea').forEach(t=>notes[t.dataset.case]=t.value);const url=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='pilot-review-notes.json';a.click();URL.revokeObjectURL(url)};</script></body></html>'''
    page=page.replace('SUMMARY',f'{len(results)} / {len(cases)} outputs ready').replace('CARDS',''.join(cards)).replace('PREFIX',json.dumps(repo.dataset_name+':'+run_id+':').replace('<','\\u003c'))
    output.write_text(page,encoding='utf-8')
    write_json_atomic(output.with_suffix('.summary.json'),report,overwrite=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='fe-pilot-v3');p.add_argument('--dataset-root',type=Path,default=Path('data/custom_datasets'))
    p.add_argument('--run-id',default='pilot-v3');p.add_argument('--output',type=Path,default=Path('analysis_outputs/perturbation_pilot/review.html'))
    a=p.parse_args();print(json.dumps(render_review(DatasetRepository.from_root(a.dataset_root,a.dataset),a.output,a.run_id),indent=2))


if __name__=='__main__': main()
