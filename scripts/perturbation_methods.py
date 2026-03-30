
#imports
from typing import Literal

from vllm import LLM, SamplingParams
import json
import sys
import torch

def apply_chat_template(base_prompt, system_prompt, ex_user, ex_assistant, context_prompt_user, text):
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": base_prompt+ex_user
        },
        {
            "role": "assistant",
            "content": ex_assistant
        },
        {
            "role": "user",
            "content": context_prompt_user+text+"\n'''\n"
        },
    ]

def select_correct_params(MODEL_PATH, ds_items, language: str, perturbation_type: Literal["clumsification", "coherence_breaking", "back_translation", "rule_based"]):
    """
    Function for determining the correct parameteres, prompts, and function calls depending on the perturbation type
    """
    #Open the correct file containing the wanted prompts
    with open("data/perturbation_prompts/"+language+"/"+perturbation_type+".json", 'r', encoding='utf-8') as reader:
        prompts = json.loads(reader.read())
    base_prompt = prompts['base_prompt']
    system_prompt = prompts['system_prompt']
    ex_user = prompts['ex_user']
    ex_assistant = prompts['ex_assistant']
    context_prompt_user = prompts['context_prompt_user']

    if perturbation_type == "clumsification":
        prompts = [apply_chat_template(base_prompt, system_prompt, ex_user, ex_assistant, context_prompt_user, x['text'].replace('\n', ' ')) for x in ds_items]
        return vllm_perturbation(MODEL_PATH, prompts)
        pass
    elif perturbation_type == "coherence_breaking":
        pass
    elif perturbation_type == "back_translation":
        pass
    #If rule-based, then functionality is completely different, as we don't use GPUs at all
    else:
        pass

def vllm_perturbation(MODEL_PATH, prompts, ):
    llm = LLM(model=MODEL_PATH, max_model_len=4096, max_num_seqs=4, gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(),)

    #Hyperparams
    pars = llm.get_default_sampling_params()
    pars.max_tokens=2048
    #pars.min_tokens=1024
    pars.temperature=0.7
    #pars.top_p=0.95
    #pars.top_k=20
    #pars.min_p=0

    outputs = llm.chat(prompts, pars)

    return outputs

def rule_based_perturbation():
    pass

def main(cmd_args):
    MODEL_PATH = cmd_args[0]
    DS_NAME = cmd_args[1]
    START_LAYER = int(cmd_args[2])
    LANGUAGE = cmd_args[3]
    PERTURBATION_TYPE = cmd_args[4]    

    ds_items = []
    ds_folder = "data/custom_datasets/"+DS_NAME+"/"
    if START_LAYER == 0:
        ds_path = ds_folder+"original.jsonl"
    else:
        ds_path = ds_folder+"perturbed-layers/"+str(START_LAYER)+".jsonl"
    
    with open(ds_path, 'r', encoding="UTF-8") as reader:
        for l in reader:
            if len(l) > 1:
                ds_items.append(json.loads(l.strip()))
    #Downsampling for testing
    ds_items = ds_items[:5]
    #Function call to have the wanted model go through the input items
    outputs = select_correct_params(MODEL_PATH, ds_items, LANGUAGE, PERTURBATION_TYPE)
    
    res_d = []
    for i,o in enumerate(outputs):
        res_d.append({'perturbation_type':PERTURBATION_TYPE ,'model':MODEL_PATH, 'head_id':i, 'text':o.outputs[0].text})

    print("Parsed outputs!")

    with open(ds_path+"perturbed_layers/"+str(START_LAYER+1)+".jsonl", "w") as writer:
        for d in res_d:
            writer.write(json.dumps(d)+'\n')
    
    print("done!")


if __name__ == "__main__":
    main(sys.argv[1:])
