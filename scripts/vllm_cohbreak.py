
#imports
from vllm import LLM, SamplingParams
import json
import sys
import torch

def apply_chat_template(base_prompt, ex_user, ex_assistant, text):
    return [
        {
            "role": "system",
            "content": "Olet opettaja, joka muokkaa teksteistä vaikeaselkoisempia, jotta oppilaat voivat oppia korjaamaan kömpelöä kieltä."
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
            "content": "Muokkaa myös seuraavaa tekstiä siten, että sen sisäinen koherenssi rikkoutuu. Älä muuta mitään erisnimiä. Jos annetussa tekstissä on kielioppivirheitä, älä korjaa niitä. Varmista, että tehdyt muokkaukset ovat kielioppisääntöjen mukaisia. Muokatun tekstin tulee olla pituudeltaan lähes sama kuin annetun tekstin. Tee mahdollisimman vähän muutoksia ja vain siten, että sisäinen koherenssi rikkoutuu. Muokkaa korkeintaan muutamaa lausetta. Faktojen täytyy olla molemmissa teksteissä samat. Palauta ainoastaan muokattu teksti ilman kommentteja tehdyistä muokkauksista. \n\nAnnettu teksti:\n'''\n"+text+"\n'''\n"
        },
    ]


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

    base_prompt = "Muokkaa annettua tekstiä siten, että sen koherenssi rikkoutuu. Muokatun tekstin tulee olla lähes sama kuin annetun tekstin. Muokatun tekstin tulee myös olla melkein yhtä pitkä, kuin annetun tekstin."
    esimerkki_1_user = "\n\nAnnettu teksti:\n'''\nMenimme eilen luokan kanssa retkelle. Ensimmäinen kohteemme oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän ja kevyen laudan, jota jokainen kantoi vuorollaan. Rakensimme sen avulla pienen sillan puron yli. Jätimme laudan metsään sellaiseen paikkaan, jonka varmasti muistamme seuraavalla retkellä.\n'''\n"
    esimerkki_1_assistant = "Muokattu teksti:\n'''\nMenimme eilen luokan kanssa retkelle. Ensimmäinen kohteemme oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän ja kevyen laudan, jota jokainen kantoi vuorollaan. Lautaan oli kirjoitettu polttomerkinnällä koulun osoite. Koulussa oli kuuma.\n'''\n"
    prompts = [apply_chat_template(base_prompt, esimerkki_1_user, esimerkki_1_assistant, x['text'].replace('\n', ' ')) for x in ds_items]

    outputs = llm.chat(prompts, pars)
    res_d = []
    for i,o in enumerate(outputs):
        res_d.append({'perturbation_type':'cohbreak' ,'model':MODEL_PATH, 'text':o.outputs[0].text, 'og_text':ds_items[i]['text']})

    print("Parsed outputs!")

    with open(output_ds_path, "w") as writer:
        for d in res_d:
            writer.write(json.dumps(d)+'\n')
    
    print("done!")


if __name__ == "__main__":
    main(sys.argv[1:])
