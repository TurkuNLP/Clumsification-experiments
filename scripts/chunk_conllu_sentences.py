from vllm import LLM
from sentence_chunk import sentence_chunking_main
import json
import os
import re
from tqdm import tqdm
import sys
import torch

def extract_sentences_from_file(file_path):
    """
    Extracts sentences from a text file where sentences are in the format "# text = {sentence}".

    Args:
        file_path (str): Path to the text file to be read.

    Returns:
        list: A list of extracted sentences.
    """
    pattern = r'#\s*text\s*=\s*(.*)'  # Regex pattern to match "# text = {sentence}"
    sentences = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(pattern, line.strip())
                if match:
                    sentences.append(match.group(1).strip()+' ')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return sentences

# Example usage:
# sentences = extract_sentences_from_file("example.txt")
# print(sentences)

def main(cmd_args):

    output_path = cmd_args[0]

    ds_items = []
    TCBC_path = "data/TCBC/"
    model = LLM(model="Qwen/Qwen3-Embedding-8B", runner="pooling", max_model_len=4096, gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(),)
    with tqdm(range(525), desc="Chunking book sentences...") as pbar:
        for f in os.listdir(TCBC_path):
            book_id = f.replace('.conllu', '')
            sentences = extract_sentences_from_file(TCBC_path+f)
        
            chunks, chunk_lens = sentence_chunking_main(512, 75, True, sentences, model)

            for i,c in enumerate(chunks):
                ds_items.append({"book_id":book_id, "chunk_id":i, "chunk_len":chunk_lens[i], "text":c})
            pbar.update(1)
    
    with open(output_path, 'w', encoding='utf-8') as writer:
        for d in ds_items:
            writer.write(json.dumps(d)+'\n')


if __name__ == "__main__":
    main(sys.argv[1:])
