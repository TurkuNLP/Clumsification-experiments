# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Generate scalar supervision JSONL files for a custom dataset.

The supported methods are ``token_normalized_perplexity``, ``bertscore_f1``,
``bleurt``, ``metricx24_source_qe``, ``gptscore_source_fluency``, and
``geval_gpt54mini_fluency``. All emit
values where higher is better. In particular,
perplexity is stored as ``-log(perplexity)`` (negative mean token NLL), because
ordinary perplexity has the opposite direction.
"""

from pathlib import Path
import sys


# Allow ``python scripts/score_custom_dataset.py ...`` from a checkout without
# requiring the repository to be installed as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clumsification_code.scoring.args import parse_score_args
from clumsification_code.scoring.custom_dataset import score_custom_dataset


def main():
    args = parse_score_args()

    result = score_custom_dataset(
        dataset_name=args.dataset_name,
        scoring_type=args.scoring_type,
        scoring_run_id=args.scoring_run_id,
        sample_limit=args.sample_limit,
        seed=args.seed,
        language=args.language,
        batch_size=args.batch_size,
        model_name=args.base_model,
        bleurt_checkpoint=args.bleurt_checkpoint,
        metricx_model_name=args.metricx_model_name,
        metricx_tokenizer_name=args.metricx_tokenizer_name,
        metricx_max_input_length=args.metricx_max_input_length,
        gptscore_model_name=args.gptscore_model_name,
        gptscore_tokenizer_name=args.gptscore_tokenizer_name,
        gptscore_model_type=args.gptscore_model_type,
        gptscore_source_prompt_template=args.gptscore_source_prompt_template,
        gptscore_device=args.gptscore_device,
        gptscore_device_map=args.gptscore_device_map,
        gptscore_dtype=args.gptscore_dtype,
        gptscore_tp_plan=args.gptscore_tp_plan,
        geval_cache_path=args.geval_cache_path,
        themis_model_name=args.themis_model_name,
        themis_tensor_parallel_size=args.themis_tensor_parallel_size,
        themis_max_model_len=args.themis_max_model_len,
        themis_max_tokens=args.themis_max_tokens,
        themis_gpu_memory_utilization=args.themis_gpu_memory_utilization,
        themis_trust_remote_code=args.themis_trust_remote_code,
        max_tokens=args.max_tokens,
        device=args.device,
        methods=args.methods,
        perturbation_run_ids=args.perturbation_run_ids,
        target_layers=args.target_layers,
        include_originals=not args.exclude_originals,
        reference_policy=args.reference_policy,
        overwrite=args.overwrite,
        dataset_root=args.dataset_root,
    )
    print(
        "Wrote {num_successful_scores} scores and {num_failures} errors.\n"
        "Scores: {score_path}\nErrors: {error_path}\nMetadata: {metadata_path}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
