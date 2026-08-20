# This script has been co-created, refactored, and cleaned using GPT 5.6.
import os
import json

def init_og_dataset(source_txt_path: str, new_ds_name: str, overwrite: bool = False):
    # If a txt-file with one text per line, then create a jsonl-file so that it fits with the rest of the code
    # If the file already contains valid JSON objects, preserve them and just add custom_id
    # Also creates the source folder for the specified dataset if necessary
    if not os.path.exists("data/custom_datasets/" + new_ds_name):
        os.makedirs("data/custom_datasets/" + new_ds_name + "/perturbed_layers", exist_ok=True)
        os.makedirs("data/custom_datasets/" + new_ds_name + "/trad_perturbed_layers", exist_ok=True)
    if not os.path.exists("data/custom_datasets/" + new_ds_name + "/original.jsonl") or overwrite:

        # Detection pass: check if every non-empty line is valid JSON
        is_jsonl = True
        has_content = False
        with open(source_txt_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                stripped = line.strip()
                if not stripped:
                    continue
                has_content = True
                try:
                    obj = json.loads(stripped)
                    if not isinstance(obj, dict):
                        is_jsonl = False
                        break
                except (json.JSONDecodeError, ValueError):
                    is_jsonl = False
                    break

        if not has_content:
            is_jsonl = False

        # Writing pass
        with open("data/custom_datasets/" + new_ds_name + "/original.jsonl", 'w', encoding='utf-8') as writer:
            with open(source_txt_path, 'r', encoding='utf-8') as reader:
                i = 0
                for line in reader:
                    stripped = line.strip()
                    if is_jsonl:
                        if not stripped:
                            continue
                        obj = json.loads(stripped)
                        if obj.get('passes_filters', None):
                            if obj['passes_filters'] == "Yes":
                                obj['custom_id'] = str(i)
                                writer.write(json.dumps(obj) + '\n')
                            else:
                                continue
                        else:
                            obj['custom_id'] = str(i)
                            writer.write(json.dumps(obj) + '\n')
                    else:
                        writer.write(json.dumps({'custom_id': str(i), 'text': line.replace('\n', '')}) + '\n')
                    i += 1

    return "data/custom_datasets/" + new_ds_name + "/original.jsonl"
