import argparse
import json
import os
import time
import traceback

import torch
import datasets
import numpy as np
from scipy.stats import spearmanr

from transformers import GPT2Tokenizer, OPTForCausalLM


# ---------------------------------------------------------------------
# Original GPTScore-style OPT scorer, kept close to the repo version.
# The repo defines GPTScore as negative LM loss on the target continuation.
# See original opt_score.py in jinlanfu/GPTScore.
# ---------------------------------------------------------------------
class OPTScorer:
    def __init__(self, device="cuda:0", max_length=1024, checkpoint=None):
        self.device = device

        # Keep GPT2Tokenizer to match the original repo's OPTScorer.
        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        self.model = OPTForCausalLM.from_pretrained(checkpoint).to(self.device)

        # Original repo overrides max_length for OPT-like models.
        max_length = 2000
        self.max_length = max_length

        print("checkpoint:", checkpoint)
        print("max_length:", max_length)

        self.model.eval()

    def score(self, srcs, tgts, prompt_text, batch_size=1):
        """
        Score examples as:

            score = log p(tgt | src + prompt_text)

        implemented as negative average target-token LM loss.

        This intentionally mirrors the original GPTScore opt_score.py logic.
        """

        def trunk_input(inputs, outputs, reduce_seq, max_length):
            input_ids = self.tokenizer.encode(inputs)[1:-1]
            output_ids = self.tokenizer.encode(outputs)[1:-1]
            reduce_seq_ids = self.tokenizer.encode(reduce_seq)[1:-1]

            total_len = len(input_ids) + len(output_ids)
            if total_len > max_length:
                del_len = total_len - max_length
                reduce_seq_ids = reduce_seq_ids[: len(reduce_seq_ids) - del_len]
                reduce_seq = self.tokenizer.decode(reduce_seq_ids[1:-1])

            return reduce_seq

        score_list = []

        for i, (src, tgt) in enumerate(zip(srcs, tgts)):
            if i % 100 == 0:
                print(f"  process {i}/{len(srcs)}")

            src = trunk_input(src, tgt, src, max_length=self.max_length)

            text = src + prompt_text + tgt

            if i == 0:
                print("  example text prefix:")
                print(text[:1000])
                print("  target:")
                print(tgt[:500])

            input_ids = self.tokenizer.encode(text)
            tgt_ids = self.tokenizer.encode(tgt)[1:]

            output_ids = [-100] * len(input_ids)
            output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids

            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            output_ids = torch.LongTensor(output_ids).unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=output_ids,
                        output_hidden_states=True,
                    )

                loss = outputs[0].item()
                score = -loss
                score_list.append(score)

            except RuntimeError:
                traceback.print_exc()
                print("input_ids:", input_ids)
                print("output_ids:", output_ids)
                print(f"source: {src}")
                print(f"target: {tgt}")

        return score_list


def detokenize(line):
    """
    Minimal local replacement.

    mteb/summeval is already readable text, so do not Moses-detokenize unless
    you intentionally want to reproduce the original preprocessing pipeline
    using their pickle files.
    """
    return line.strip()


def add_dot(line):
    """
    Close to the original repo's add_dot idea: add final punctuation.

    The original utils.add_dot adds ' .' if final char is not '.'. Here we use
    normal punctuation to avoid damaging already-normal Hugging Face text.
    """
    line = line.strip()
    if not line:
        return line

    if line[-1] not in [".", "?", "!", '"', "'"]:
        line += "."

    return line


# ---------------------------------------------------------------------
# SummEval src->hypo prompt definitions from GPTScore Table 11.
#
# The full prompt seen by OPT is:
#
#   source_side + "\n\nTl;dr" + system_summary
#
# For IST, source_side already contains:
#
#   "Generate a coherent summary for the following text: {src}"
#
# For VAL, source_side is just:
#
#   "{src}"
# ---------------------------------------------------------------------
SUMMEVAL_ASPECTS = {
    "coherence": {
        "abbr": "COH",
        "instruction_prefix": "Generate a coherent summary for the following text: ",
    },
    "consistency": {
        "abbr": "CON",
        "instruction_prefix": "Generate factually consistent summary for the following text: ",
    },
    "fluency": {
        "abbr": "FLU",
        "instruction_prefix": "Generate a fluent and grammatical summary for the following text: ",
    },
    "relevance": {
        "abbr": "REL",
        "instruction_prefix": "Generate a relevant summary with consistent details for the following text: ",
    },
}


class SummEvalOPT350MScorer:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        print("Loading SummEval from Hugging Face: mteb/summeval")
        self.raw_ds = datasets.load_dataset("mteb/summeval")["test"]

        # Keep data nested by source document so sample-level Spearman is easy.
        self.data = self._build_data()

        print("num source documents:", len(self.data))
        print("num summaries:", sum(len(x["systems"]) for x in self.data))

    def _build_data(self):
        data = []

        for doc_idx, row in enumerate(self.raw_ds):
            systems = []

            machine_summs = row["machine_summaries"]

            for sys_idx, sys_summ in enumerate(machine_summs):
                systems.append(
                    {
                        "sys_idx": sys_idx,
                        "sys_summ": add_dot(detokenize(sys_summ)),
                        "annotations": {
                            "coherence": float(row["coherence"][sys_idx]),
                            "consistency": float(row["consistency"][sys_idx]),
                            "fluency": float(row["fluency"][sys_idx]),
                            "relevance": float(row["relevance"][sys_idx]),
                        },
                        "scores": {},
                    }
                )

            data.append(
                {
                    "doc_idx": doc_idx,
                    "id": row.get("id", str(doc_idx)),
                    "source_text": add_dot(detokenize(row["text"])),
                    "systems": systems,
                }
            )

        return data

    def save_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        print(f"Saved scored data to: {path}")

    def _source_side(self, source_text, aspect_name, instruction, no_reference):
        """
        Build the string before '\n\nTl;dr'.

        Main paper replication:
            no_reference=False
            source_text = original CNN/DM article

        No-reference sanity test:
            no_reference=True
            source_text replaced by exactly ' '
        """
        src = " " if no_reference else source_text

        if instruction:
            prefix = SUMMEVAL_ASPECTS[aspect_name]["instruction_prefix"]
            return prefix + src

        return src

    def score_all(self):
        checkpoint = "facebook/opt-350m"
        opt_scorer = OPTScorer(device=self.device, checkpoint=checkpoint)

        prompt_text = "\n\nTl;dr"

        settings = [
            # Paper replication settings.
            {
                "name": "opt350m_val_src_hypo",
                "instruction": False,
                "no_reference": False,
            },
            {
                "name": "opt350m_ist_src_hypo",
                "instruction": True,
                "no_reference": False,
            },
            # Your requested no-reference sanity checks.
            # Only difference: source text is replaced by " ".
            {
                "name": "opt350m_val_noref_hypo",
                "instruction": False,
                "no_reference": True,
            },
            {
                "name": "opt350m_ist_noref_hypo",
                "instruction": True,
                "no_reference": True,
            },
        ]

        start_all = time.time()

        for setting in settings:
            setting_name = setting["name"]
            instruction = setting["instruction"]
            no_reference = setting["no_reference"]

            print()
            print("=" * 80)
            print(f"Scoring setting: {setting_name}")
            print("=" * 80)

            # For VAL, the metric score is aspect-independent.
            # For IST, prompts are aspect-specific.
            aspects_to_score = (
                list(SUMMEVAL_ASPECTS.keys())
                if instruction
                else ["__vanilla__"]
            )

            for aspect_name in aspects_to_score:
                if aspect_name == "__vanilla__":
                    score_key = setting_name
                    display_aspect = "vanilla"
                else:
                    score_key = f"{setting_name}_{aspect_name}"
                    display_aspect = aspect_name

                print()
                print(f"Aspect/prompt: {display_aspect}")
                start = time.time()

                srcs = []
                tgts = []
                index = []

                for doc_idx, doc in enumerate(self.data):
                    for sys_idx, system in enumerate(doc["systems"]):
                        if aspect_name == "__vanilla__":
                            # Aspect does not matter for vanilla.
                            source_side = " " if no_reference else doc["source_text"]
                        else:
                            source_side = self._source_side(
                                source_text=doc["source_text"],
                                aspect_name=aspect_name,
                                instruction=instruction,
                                no_reference=no_reference,
                            )

                        srcs.append(source_side)
                        tgts.append(system["sys_summ"])
                        index.append((doc_idx, sys_idx))

                scores = opt_scorer.score(
                    srcs=srcs,
                    tgts=tgts,
                    prompt_text=prompt_text,
                    batch_size=self.args.batch_size,
                )

                if len(scores) != len(index):
                    raise RuntimeError(
                        f"Expected {len(index)} scores, got {len(scores)}"
                    )

                for (doc_idx, sys_idx), score in zip(index, scores):
                    self.data[doc_idx]["systems"][sys_idx]["scores"][score_key] = float(score)

                print(
                    f"Finished {score_key}; "
                    f"time passed {time.time() - start:.1f}s"
                )

        print(f"All scoring finished; total time {time.time() - start_all:.1f}s")

        del opt_scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate_sample_level_spearman(self):
        """
        Replicate paper-style summarization evaluation:

            For each source document:
                Spearman(metric scores over 16 systems,
                         human scores over 16 systems)

            Then average across source documents.

        Paper reports Spearman * 100.
        """
        print()
        print("=" * 80)
        print("Sample-level Spearman, averaged over source documents")
        print("Paper reports these values multiplied by 100.")
        print("=" * 80)

        result = {}

        # Evaluation score keys:
        # VAL scores are shared across aspects.
        val_keys = [
            "opt350m_val_src_hypo",
            "opt350m_val_noref_hypo",
        ]

        ist_keys = []
        for setting_prefix in [
            "opt350m_ist_src_hypo",
            "opt350m_ist_noref_hypo",
        ]:
            for aspect_name in SUMMEVAL_ASPECTS:
                ist_keys.append(f"{setting_prefix}_{aspect_name}")

        score_keys = val_keys + ist_keys

        for score_key in score_keys:
            if "ist" in score_key:
                # For instruction, each score key corresponds naturally to one aspect.
                candidate_aspects = [
                    a for a in SUMMEVAL_ASPECTS if score_key.endswith("_" + a)
                ]
            else:
                # Vanilla/no-ref vanilla score can be correlated against all aspects.
                candidate_aspects = list(SUMMEVAL_ASPECTS.keys())

            print()
            print(f"Score key: {score_key}")
            result[score_key] = {}

            for ann_key in candidate_aspects:
                per_doc_corrs = []

                for doc in self.data:
                    metric_scores = []
                    human_scores = []

                    for system in doc["systems"]:
                        if score_key not in system["scores"]:
                            raise KeyError(f"Missing score: {score_key}")

                        metric_scores.append(system["scores"][score_key])
                        human_scores.append(system["annotations"][ann_key])

                    corr = spearmanr(metric_scores, human_scores).correlation

                    if corr is not None and not np.isnan(corr):
                        per_doc_corrs.append(corr)

                avg_corr = float(np.mean(per_doc_corrs))
                result[score_key][ann_key] = avg_corr

                print(
                    f"  {ann_key:12s}: "
                    f"{avg_corr:.4f}  "
                    f"paper-style x100 = {100.0 * avg_corr:.1f}"
                )

        return result


def main():
    parser = argparse.ArgumentParser(
        description="SummEval GPTScore replication: OPT-350M src->hypo"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device, e.g. cuda:0 or cpu.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Kept for API compatibility; original OPTScorer scores one by one.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/summeval_opt350m_src_hypo_and_noref.json",
        help="Path to save calculated scores.",
    )

    args = parser.parse_args()

    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))

    scorer = SummEvalOPT350MScorer(args)
    scorer.score_all()
    scorer.evaluate_sample_level_spearman()
    scorer.save_data(args.output)


if __name__ == "__main__":
    main()