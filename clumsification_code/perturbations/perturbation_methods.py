# This script has been co-created, refactored, and cleaned using GPT 5.6.
# imports
import argparse
import json
import random
import re
import sys
import torch
import gc
from math import floor
from typing import Literal

from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

from transformers import AutoTokenizer
import numpy as np

import os
from clumsification_code.perturbations.evaleval_perturbations import (
    RULE_BASED_MODEL_LABEL,
    RULE_BASED_OUTPUT_DIR,
    rule_based_perturbation,
)

VALID_PERTURBATION_TYPES = (
    "clumsification",
    "coherence_breaking",
    "back_translation",
    "rule_based",
)

# Estimated token overhead for the full chat template (system prompt,
# few-shot example turn, context instruction, boilerplate text).
# Measured once and hardcoded — bump this up if you add longer prompts.
PROMPT_OVERHEAD_TOKENS = 512

# Bucket boundaries (in *text* tokens, before prompt overhead).
# Items are placed in the smallest bucket that fits their text.
BUCKET_BOUNDARIES = [512, 1024, 2048, 4096, 8192, 16384]



# Bucket-parameter helpers


def _next_power_of_two(n):
    p = 1
    while p < n:
        p *= 2
    return p


def compute_bucket_params(bucket_upper_bound):
    """
    Given the upper text-token bound for a bucket, return
    (max_model_len, max_tokens, thinking_token_budget).

    Layout inside the context window:
        [prompt overhead + text tokens]  →  input
        [thinking tokens + rewritten text tokens]  →  output
        total ≈ PROMPT_OVERHEAD + B + thinking + B + small buffer

    """
    B = bucket_upper_bound

    # First pass: estimate thinking budget as ~45 % of text length
    thinking_budget = max(1024, int(B * 0.45))

    total_estimate = PROMPT_OVERHEAD_TOKENS + B + B + thinking_budget + 128

    # If we'd exceed 4k, cap thinking at 4096 and recompute
    if total_estimate > 4000:
        thinking_budget = 4096
        total_estimate = PROMPT_OVERHEAD_TOKENS + B + B + thinking_budget + 128

    max_model_len = _next_power_of_two(total_estimate)

    # max_tokens (SamplingParams): enough room for rewritten text + thinking
    max_tokens = B + thinking_budget + 256

    return max_model_len, max_tokens, thinking_budget


def assign_buckets(ds_items):
    """
    Partition *ds_items* into mutually-exclusive length buckets.
    Each item goes into the smallest bucket whose boundary >= item's
    text token length.  Items that exceed all boundaries go into an
    overflow bucket whose boundary is the next power of two above the
    longest item.

    Returns
    -------
    dict[int, list]   –  {bucket_boundary: [items …]}
    """
    buckets = {}
    for item in ds_items:
        tl = item["_token_length"]
        placed = False
        for boundary in BUCKET_BOUNDARIES:
            if tl <= boundary:
                buckets.setdefault(boundary, []).append(item)
                placed = True
                break
        if not placed:
            # Overflow: create a bucket at the next power of two
            overflow = _next_power_of_two(tl)
            buckets.setdefault(overflow, []).append(item)
    return buckets

def merge_buckets_by_model_len(buckets):
    """
    Merge length-buckets that would produce the same max_model_len
    into a single group, so vLLM is only initialised once per distinct
    context-window size.

    Parameters
    ----------
    buckets : dict[int, list]
        {bucket_boundary: [items …]}  as returned by assign_buckets()

    Returns
    -------
    list[dict]  –  one entry per unique max_model_len, sorted ascending:
        {
            "max_model_len":  int,
            "max_tokens":     int,   # max across merged buckets
            "thinking_budget": int,  # max across merged buckets
            "items":          list,
            "label":          str,   # human-readable merged-boundary label
        }
    """
    # Step 1: compute params for every raw bucket and group by max_model_len
    groups = {}  # max_model_len → {boundaries, items, max_tokens, thinking}
    for boundary in sorted(buckets):
        items = buckets[boundary]
        if not items:
            continue
        mml, mt, tb = compute_bucket_params(boundary)
        if mml not in groups:
            groups[mml] = {
                "max_model_len": mml,
                "max_tokens": mt,
                "thinking_budget": tb,
                "items": [],
                "boundaries": [],
            }
        g = groups[mml]
        g["items"].extend(items)
        g["boundaries"].append(boundary)
        # Take the most generous output settings so the largest items fit
        g["max_tokens"] = max(g["max_tokens"], mt)
        g["thinking_budget"] = max(g["thinking_budget"], tb)

    # Step 2: build a nice label and return sorted by max_model_len
    merged = []
    for mml in sorted(groups):
        g = groups[mml]
        bounds = g["boundaries"]
        if len(bounds) == 1:
            label = f"<={bounds[0]}"
        else:
            label = "+".join(f"<={b}" for b in bounds)
        merged.append(
            {
                "max_model_len": g["max_model_len"],
                "max_tokens": g["max_tokens"],
                "thinking_budget": g["thinking_budget"],
                "items": g["items"],
                "label": label,
            }
        )
    return merged

# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text perturbation using a vLLM model or rule-based methods."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model used for perturbation.",
    )
    parser.add_argument(
        "--ds-names",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset names to load, merge, and shuffle.",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        required=True,
        help="Layer index. 0 → original.jsonl, else perturbed_layers/<N>.jsonl.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code used to locate prompt files (e.g. 'en').",
    )
    parser.add_argument(
        "--perturbation-type",
        type=str,
        required=True,
        choices=VALID_PERTURBATION_TYPES,
        help="Type of perturbation to apply.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: keep only the first N items (for testing).",
    )
    parser.add_argument(
        "--rule-task",
        choices=["all", "MT", "IC", "AS", "D2T", "QG", "DG", "COMMON_FLUENCY"],
        default="all",
    )
    parser.add_argument("--rule-criteria", default="all")
    parser.add_argument("--rule-templates", nargs="+", default=None)
    parser.add_argument("--rule-output-mode", choices=["all", "first_success", "random_success"], default="all")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()



# Prompt / template helpers


def apply_chat_template(
    base_prompt, system_prompt, ex_user, ex_assistant,
    context_prompt_user, text, max_length,
):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_prompt + ex_user},
        {"role": "assistant", "content": ex_assistant},
        {
            "role": "user",
            "content": (
                context_prompt_user
                + "\n The absolute maximum length of the edited text in characters is "
                + str(max_length)
                + ". You are never allowed to edit a text to be longer than this."
                + " Now, edit this text: \n"
                + text
            ),
        },
    ]


def build_prompts(ds_items, language, perturbation_type):
    """
    Build the list of chat-message prompts for the given perturbation type.
    Returns a list parallel to *ds_items*.
    """
    prompt_path = (
        "data/perturbation_prompts/"
        + language
        + "/"
        + perturbation_type
        + ".json"
    )
    with open(prompt_path, "r", encoding="utf-8") as reader:
        prompts = json.loads(reader.read())

    base_prompt = prompts["base_prompt"]
    system_prompt = prompts["system_prompt"]
    ex_user = prompts["ex_user"]
    ex_assistant = prompts["ex_assistant"]
    context_prompt_user = prompts["context_prompt_user"]

    if perturbation_type == "clumsification":
        return [
            apply_chat_template(
                base_prompt,
                system_prompt,
                ex_user,
                ex_assistant,
                context_prompt_user,
                x["text"].replace("\n", " "),
                x["max_length"],
            )
            for x in ds_items
        ]
    elif perturbation_type == "coherence_breaking":
        raise NotImplementedError("coherence_breaking is not yet implemented")
    elif perturbation_type == "back_translation":
        raise NotImplementedError("back_translation is not yet implemented")
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")



# vLLM inference (one bucket at a time)


def vllm_perturbation(
    model_path, prompts, max_model_len, max_tokens, thinking_token_budget,
):
    """
    Spin up a vLLM instance sized for this bucket, run inference,
    then tear it down so the next bucket can use different settings.
    """
    reasoning_config = ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str=(
            "I have finished my analysis. I will now give the final answer only."
            "</think>"
        ),
    )

    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        tensor_parallel_size=torch.cuda.device_count(),
        language_model_only=True,
        reasoning_config=reasoning_config,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        thinking_token_budget=thinking_token_budget,
    )

    outputs = llm.chat(
        messages=prompts,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": True},
    )

    # Free GPU memory before the next bucket's LLM is created
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return outputs

# Output parsing helpers


def get_text_after_last_think(text: str) -> str:
    tag = "</think>"
    last_index = text.rfind(tag)
    if last_index == -1:
        return text
    return text[last_index + len(tag) :]



# Dataset loading


def load_dataset_items(ds_name, pert_type, start_layer):
    ds_folder = "data/custom_datasets/" + ds_name + "/"
    if start_layer == 0:
        ds_path = ds_folder + "original.jsonl"
    else:
        if pert_type == "rule_based":
            ds_path = ds_folder + "trad_perturbed_layers/" + str(start_layer) + ".jsonl"
        else:
            ds_path = ds_folder + "perturbed_layers/" + str(start_layer) + ".jsonl"

    items = []
    with open(ds_path, "r", encoding="UTF-8") as reader:
        for i,line in enumerate(reader):
            if len(line.strip()) > 0:
                item = json.loads(line.strip())
                item["_source_ds"] = ds_name
                item["_source_index"] = i
                items.append(item)
    return items, ds_folder



# Main


def main():
    args = parse_args()

    MODEL_PATH = args.model_path
    DS_NAMES = args.ds_names
    START_LAYER = args.start_layer
    LANGUAGE = args.language
    PERTURBATION_TYPE = args.perturbation_type
    DOWNSAMPLE = args.limit

    
    # Load, merge datasets
    
    ds_items = []
    ds_folders = {}
    for ds_name in DS_NAMES:
        items, folder = load_dataset_items(ds_name, PERTURBATION_TYPE, START_LAYER)
        ds_folders[ds_name] = folder
        ds_items.extend(items)

    if START_LAYER == 0:
        for i, _ in enumerate(ds_items):
            ds_items[i]["max_length"] = min(
                floor(len(ds_items[i]["text"].replace("\n", " ")) * 1.1),
                len(ds_items[i]["text"].replace("\n", " ")) + 500,
            )

    # Optional downsampling
    if DOWNSAMPLE is not None:
        ds_items = ds_items[:DOWNSAMPLE]

    
    # Rule-based: no LLM needed — skip tokenisation & bucketing entirely

    for i, item in enumerate(ds_items):
        item["_original_index"] = i
        if "max_length" not in item:
            clean_text = item["text"].replace("\n", " ")
            item["max_length"] = min(floor(len(clean_text) * 1.1), len(clean_text) + 500)
    
    if PERTURBATION_TYPE == "rule_based":
        random.seed(args.seed)
        res_d = rule_based_perturbation(
            ds_items,
            rule_task=args.rule_task,
            rule_criteria=args.rule_criteria,
            rule_template_names=args.rule_templates,
            output_mode=args.rule_output_mode,
            model_label=MODEL_PATH or RULE_BASED_MODEL_LABEL,
        )
        output_layer = str(START_LAYER + 1)
        for ds_name in DS_NAMES:
            out_dir = os.path.join(ds_folders[ds_name], RULE_BASED_OUTPUT_DIR)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, output_layer + ".jsonl")
            subset = [d for d in res_d if d["_source_ds"] == ds_name]
            with open(out_path, "w", encoding="UTF-8") as writer:
                for d in subset:
                    row = {k: v for k, v in d.items() if k != "_source_ds"}
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Wrote {len(subset)} items to {out_path}")
        print("Done!")
        return

    
    # Tokenize all items and collect token lengths
    
    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Tokenizing {len(ds_items)} items ...")
    token_lengths = []
    for i, item in enumerate(ds_items):
        token_ids = tokenizer.encode(item["text"], add_special_tokens=False)
        tl = len(token_ids)
        token_lengths.append(tl)
        ds_items[i]["_token_length"] = tl

    token_lengths = np.array(token_lengths)

    
    # Print token-length statistics
    
    print("\n" + "=" * 60)
    print("TOKEN LENGTH STATISTICS")
    print("=" * 60)
    print(f"  Total items:         {len(token_lengths)}")
    print(f"  Min token length:    {token_lengths.min()}")
    print(f"  Max token length:    {token_lengths.max()}")
    print(f"  Mean token length:   {token_lengths.mean():.1f}")
    print(f"  Median token length: {np.median(token_lengths):.1f}")
    print(f"  Std dev:             {token_lengths.std():.1f}")

    print("\n  Percentile breakdown:")
    for p in [50, 75, 90, 95, 99, 99.5, 100]:
        val = np.percentile(token_lengths, p)
        count = int(np.sum(token_lengths <= val))
        pct = count / len(token_lengths) * 100
        print(
            f"    P{p:<5} = {val:>7.0f} tokens  "
            f"({count}/{len(token_lengths)} items, {pct:.1f}%)"
        )

    print("\n  Items fitting within common context lengths:")
    for b in BUCKET_BOUNDARIES:
        count = int(np.sum(token_lengths <= b))
        pct = count / len(token_lengths) * 100
        print(f"    <= {b:>6} tokens:  {count:>6} items  ({pct:>5.1f}%)")
    print("=" * 60 + "\n")

    

    # Assign items to length buckets, then merge by max_model_len

    raw_buckets = assign_buckets(ds_items)

    print("Raw bucket plan (before merging):")
    for boundary in sorted(raw_buckets):
        mml, mt, tb = compute_bucket_params(boundary)
        print(
            f"  <= {boundary:>6} text tokens : {len(raw_buckets[boundary]):>6} items  │  "
            f"max_model_len={mml:>6}  max_tokens={mt:>6}  "
            f"thinking_budget={tb:>5}"
        )

    merged_groups = merge_buckets_by_model_len(raw_buckets)

    print(f"\nMerged into {len(merged_groups)} vLLM instance(s):")
    for g in merged_groups:
        print(
            f"  {g['label']:>40s} : {len(g['items']):>6} items  │  "
            f"max_model_len={g['max_model_len']:>6}  "
            f"max_tokens={g['max_tokens']:>6}  "
            f"thinking_budget={g['thinking_budget']:>5}"
        )
    print()


    # Process each merged group: build prompts → spin up vLLM → infer

    all_results = []  # list of (item_dict, vllm_output) pairs

    for g in merged_groups:
        bucket_items = g["items"]
        if not bucket_items:
            continue

        print(
            f"▶ Group {g['label']}: {len(bucket_items)} items  |  "
            f"max_model_len={g['max_model_len']}  "
            f"max_tokens={g['max_tokens']}  "
            f"thinking_budget={g['thinking_budget']}"
        )

        # Shuffle within the group so items from different datasets are
        # interleaved (avoids systematic ordering effects)
        random.shuffle(bucket_items)

        # Build chat prompts for this group
        chat_prompts = build_prompts(bucket_items, LANGUAGE, PERTURBATION_TYPE)

        # Run inference
        outputs = vllm_perturbation(
            MODEL_PATH,
            chat_prompts,
            max_model_len=g["max_model_len"],
            max_tokens=g["max_tokens"],
            thinking_token_budget=g["thinking_budget"],
        )

        # Collect results
        for item, output in zip(bucket_items, outputs):
            all_results.append((item, output))

        print(f"  ✓ Finished group {g['label']}\n")

    
    # Restore original ordering (so head_id stays consistent)
    
    all_results.sort(key=lambda pair: pair[0]["_original_index"])

    
    # Parse outputs
    
    if not all_results:
        print("ERROR: no outputs were produced.", file=sys.stderr)
        sys.exit(1)

    res_d = []
    for item, o in all_results:
        temp_text = o.outputs[0].text
        temp_text = re.sub(r"<think>.*?</think>", "", temp_text, flags=re.DOTALL)
        temp_text = re.sub(r"\A[\n']+|[\n']+\Z", "", temp_text)
        temp_text = get_text_after_last_think(temp_text)
        res_d.append(
            {
                "perturbation_type": PERTURBATION_TYPE,
                "model": MODEL_PATH,
                "head_id": item.get("_source_index", item["_original_index"]),
                "text": temp_text,
                "max_length": item["max_length"],
                "_source_ds": item["_source_ds"],
            }
        )

    print(f"Parsed {len(res_d)} outputs!")

    
    # Write results — one output file per source dataset
    
    output_layer = str(START_LAYER + 1)
    for ds_name in DS_NAMES:
        out_path = ds_folders[ds_name] + "perturbed_layers/" + output_layer + ".jsonl"
        subset = [d for d in res_d if d["_source_ds"] == ds_name]
        with open(out_path, "w", encoding="UTF-8") as writer:
            for d in subset:
                row = {k: v for k, v in d.items() if k != "_source_ds"}
                writer.write(json.dumps(row) + "\n")
        print(f"Wrote {len(subset)} items to {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
