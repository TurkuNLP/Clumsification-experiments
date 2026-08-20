# This script has been co-created, refactored, and cleaned using GPT 5.6.
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig
import json
import sys
import torch
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen thinking-mode LLM inference over a JSONL dataset."
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path or identifier of the model to load.",
    )

    parser.add_argument(
        "--ds-name",
        required=True,
        help="Path to the input JSONL dataset.",
    )

    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where the output JSONL file will be written.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of dataset examples to process.",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help=(
            "Maximum model context length. Must be large enough for the prompt "
            "plus generated thinking/final-answer tokens."
        ),
    )

    parser.add_argument(
        "--thinking-token-budget",
        type=int,
        default=2048,
        help="Maximum number of thinking tokens to allow before final answer.",
    )

    parser.add_argument(
        "--max-final-answer-tokens",
        type=int,
        default=32,
        help="Extra tokens reserved after thinking for final Yes/No answer.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 gives deterministic classification.",
    )

    parser.add_argument(
        "--save-raw-output",
        action="store_true",
        help="If set, also save the raw model output including thinking text.",
    )

    return parser.parse_args()


def apply_chat_template(given_example: str):

    """
    Function for applying a chat template to use in LLM inference
    """

    filter_prompt = """
    You are tasked with deciding whether a text sohuld get accepted to be studied by scientists working on evaluating text quality or filtered out
    .
    You should filter out texts using these criteria:
    1. A text should be filtered if it mainly consists of very spoken or informal language
    2. A text should be filtered if it mostly consists of lists or listing properties akin to and advert instead of proper sentences
    3. A text should be filtered if it is not at least four sentences long
    4. A text should be filtered if it contains many headers, timestamps, links or other elements that are typical in online websites, but not in news articles or book chapters
    5. A text should be filtered if it contains a high ratio of non-alphabetical characters (>85%), such as numbers, dashes or colons

    Return your answer ONLY as Yes or No.
    Answer Yes only if a text does not meet any of the criteria for filtering.
    In difficult or debatable cases, you should answer Yes.

    You may think internally. However, after your thinking is complete, your final answer must be exactly one word:

    Yes

    or

    No

    Do not include anything else in the final answer.
    """.strip()

    example_1 = """
    Here is a text you need to analyze and decide whether it should be accepted or not:

    Audio music recordings of traditional music plus historical, ethnographic and government documents, maps, photographs, tapes, artifacts and memorabilia. From the Holsoe, d'Azevedo and several smaller collections.. Resource for the documentary film 'Liberia: America's Stepchild' provides a timeline, map, glossary, essays, lesson plans, educators' resources and links for teachers, students, and the general public.
    """.strip()

    reply_1 = """
    No
    """.strip()

    example_2 = """
    Now analyze this following text and decide whether it should be accepted or not:

    Aaron Schatz of Football Outsiders joins me for a wide ranging discussion on the NFL. He tells us about the most important frontiers in football analytics, which leads to a discussion of which defensive stats are least stable from year to year. Then we get into an overrated team, or Jacksonville 2.0. Finally, he tells us about an NFC North team that might surprise you.
    """.strip()

    reply_2 = """
    Yes
    """.strip()

    user_1 = """
    Now, finally, analyze this following text and whether it should be accepted or not.

    Remember:
    - You may think internally.
    - The final answer after thinking must be exactly one word: Yes or No.
    - Do not include explanations in the final answer.

    Text to analyze:

    """


    return [
        {
            "role": "system",
            "content": filter_prompt
        },
        {
            "role": "user",
            "content": example_1
        },
        {
            "role": "assistant",
            "content": reply_1
        },
        {
            "role": "user",
            "content": example_2
        },
        {
            "role": "assistant",
            "content": reply_2
        },
        {
            "role": "user",
            "content": user_1+given_example
        },
    ]

def load_jsonl(path, limit=None):
    data = []

    with open(path, "r", encoding="utf-8") as reader:
        for line_num, line in enumerate(reader, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e

            if limit is not None and len(data) >= limit:
                break

    return data


def clean_yes_no(text: str) -> str:
    """
    Extract final Yes/No from Qwen thinking-mode output.

    Qwen thinking output often looks like:

        ...reasoning...
        </think>
        Yes

    For offline vLLM, the separate reasoning field is not generally available,
    so we parse the raw text ourselves.

    Strategy:
    1. If </think> exists, only inspect text after the last </think>.
    2. Remove any leftover think blocks defensively.
    3. Prefer a standalone Yes/No on the final non-empty line.
    4. Fall back to the last standalone Yes/No.
    """

    if text is None:
        return "INVALID"

    raw = text.strip()

    # If Qwen produced a thinking block, ignore everything before the final </think>.
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1].strip()

    # Defensive cleanup in case a full think block remains.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Inspect final non-empty line first.
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        final_line = lines[-1]
        if re.fullmatch(r"(?i)yes[.!]?", final_line):
            return "Yes"
        if re.fullmatch(r"(?i)no[.!]?", final_line):
            return "No"

    # Fallback: find the last standalone Yes/No after </think>.
    matches = re.findall(r"\b(Yes|No)\b", raw, flags=re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()

    return "INVALID"



def main():
    args = parse_args()

    print("Arrived")

    ds_loaded = load_jsonl(path=args.ds_name, limit=args.limit)

    prompts = []
    for x in ds_loaded:
        prompts.append(apply_chat_template(x['text']))

    print("Loaded")

    reas_conf = ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="I have to give the solution based on the reasoning directly now.</think>"
    )
    


    tensor_parallel_size = torch.cuda.device_count()

    if tensor_parallel_size == 0:
        raise RuntimeError(
            "No CUDA devices found. This script expects at least one GPU."
        )

    # Qwen uses <think> ... </think>. The custom reasoning_end_str makes budget
    # exhaustion transition cleanly into the final answer.
    reasoning_config = ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str=(
            "I have finished my analysis. I will now give the final answer only."
            "</think>"
        ),
    )

    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        language_model_only=True,
        reasoning_config=reasoning_config,
    )

    max_tokens = args.thinking_token_budget + args.max_final_answer_tokens

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=args.temperature,
        thinking_token_budget=args.thinking_token_budget,
    )

    outputs = llm.chat(
        messages=prompts,
        sampling_params=sampling_params,
        chat_template_kwargs={
            "enable_thinking": True,
        },
    )

    to_write = []
    for i,o in enumerate(outputs):
        temp_text = o.outputs[0].text
        temp_text = re.sub(r'Thinking Process:\n\n.*?</think>', '', temp_text, flags=re.DOTALL)
        temp_text = re.sub(r"\A[\n']+|[\n']+\Z", '', temp_text)
        to_write.append({
            'text':ds_loaded[i]['text'],
            'warc_id':ds_loaded[i]['warc_id'],
            'passes_filters':temp_text
        })

    with open(args.output_path, "w", encoding="utf-8") as writer:
        for x in to_write:
            writer.write(json.dumps(x, ensure_ascii=False) + "\n")

if __name__ == "__main__":
     main()
