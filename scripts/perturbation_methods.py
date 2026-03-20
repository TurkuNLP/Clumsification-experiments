
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

def select_correct_params(perturbation_type: Literal["clumsification", "coh_breaking", "back_translation", "rule_based"]):
    """
    Function for determining the correct parameteres, prompts, and function calls depending on the perturbation type
    """

    base_prompt = ""
    system_prompt = ""
    ex_user = ""
    ex_assistant = ""
    context_prompt_user = ""

    if perturbation_type == "clumsification":
        base_prompt = "Muokkaa annetusta tektistä kömpelömpi ja vaikeaselkoisempi. Muokatun tekstin tulee olla lähes sama kuin annetun tekstin ja asioiden täytyy esiintyä molemmissa teksteissä samassa järjestyksessä. Muokatun tekstin tulee myös olla melkein yhtä pitkä, kuin annetun tekstin."
        system_prompt = "Olet opettaja, joka muokkaa teksteistä vaikeaselkoisempia, jotta oppilaat voivat oppia korjaamaan kömpelöä kieltä."
        ex_user = "\n\nAnnettu teksti:\n'''\nMenimme eilen luokan kanssa retkelle. Ensimmäinen kohteemme oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän ja kevyen laudan, jota jokainen kantoi vuorollaan. Rakensimme sen avulla pienen sillan puron yli. Jätimme laudan metsään sellaiseen paikkaan, jonka varmasti muistamme seuraavalla retkellä.\n'''\n"
        ex_assistant = "Muokattu teksti:\n'''\nEilen menimme luokan kanssa retkelle, ja ensimmäinen paikka oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän esineen nimeltä lauta, joka oli niin kevyt, että jokainen jaksoi kantaa sitä vuorollaan. Rakensimme laudan avulla pienen sillan puron yli, ja se jäi metsään paikalle, jonka muistamme varmasti seuraavalla retkellä.\n'''\n"
        context_prompt_user = "Muokkaa myös seuraavasta tekstistä kömpelömpi. Älä muuta mitään erisnimiä tai kommentoi tehtyjä muokkauksia. Jos annetussa tekstissä on kielioppivirheitä, älä korjaa niitä. Varmista, että tehdyt muokkaukset tekevät tekstistä vaikeammin luettavan, mutta ovat kielioppisääntöjen mukaisia. \n\nAnnettu teksti:\n'''\n"


        pass
    elif perturbation_type == "coh_breaking":
        pass
    elif perturbation_type == "back_translation":
        pass
    #If rule-based, then functionality is completely different, as we don't use GPUs at all
    else:
        pass

def vllm_perturbation():
    pass

def rule_based_perturbation():
    pass

def main(cmd_args):
    MODEL_PATH = cmd_args[0]
    llm = LLM(model=MODEL_PATH, max_model_len=4096, max_num_seqs=4, gpu_memory_utilization=0.9, tensor_parallel_size=torch.cuda.device_count(),)
    source_ds_path = cmd_args[1]
    output_ds_path = cmd_args[2]
    ds_items = []
    with open(source_ds_path, 'r', encoding="UTF-8") as reader:
        for l in reader:
            if len(l) > 1:
                ds_items.append(json.loads(l.strip()))
    ds_items = ds_items[14470:14505]
    pars = llm.get_default_sampling_params()
    pars.max_tokens=2048
    #pars.min_tokens=1024
    pars.temperature=0.7
    #pars.top_p=0.95
    #pars.top_k=20
    #pars.min_p=0

    prompts = [apply_chat_template(base_prompt, esimerkki_1_user, esimerkki_1_assistant, x['text'].replace('\n', ' ')) for x in ds_items]

    outputs = llm.chat(prompts, pars)
    res_d = []
    for i,o in enumerate(outputs):
        res_d.append({'perturbation_type':'clumsification' ,'model':MODEL_PATH, 'text':o.outputs[0].text, 'og_text':ds_items[i]['text']})

    print("Parsed outputs!")

    with open(output_ds_path, "w") as writer:
        for d in res_d:
            writer.write(json.dumps(d)+'\n')
    
    print("done!")


if __name__ == "__main__":
    main(sys.argv[1:])
