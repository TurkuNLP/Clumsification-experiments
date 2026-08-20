# This script has been co-created, refactored, and cleaned using GPT 5.6.

#Imports
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import json
import sys
import os
from pprint import pprint
import multiprocessing as mp
from tqdm import tqdm
import clumsification_code.perturbations.OpenAI_lib as ol
import time
from openai.lib._parsing._responses import type_to_text_format_param
from openai.types.responses import Response
from openai.lib._parsing._responses import parse_response

client = ol.get_client_local()

BASE_FOLDER = "data/custom_datasets/"
PERTURBATION_FOLDER = "perturbed_layers/"


#Define wished output structure and method for prompting GPT
class EditedTexts(BaseModel):
    text: str

def prompt_gpt(model, user_prompt, reasoning_effort, format):
    completion = client.responses.parse(
            model=model,
            input=user_prompt,
            reasoning={ "effort": reasoning_effort},
            text_format=format,
        )
    return completion.output_parsed

def generate_batch_item(og_dataset_name, batch_id, text, model_name, effort, text_format, language, perturbation_type):
    bitem = {"custom_id":str(batch_id), "method":"POST", "url":"/v1/responses"}
    #Open the correct file containing the wanted prompts
    with open("data/perturbation_prompts/"+language+"/"+perturbation_type+".json", 'r', encoding='utf-8') as reader:
        prompts = json.loads(reader.read())
    base_prompt = prompts['base_prompt']
    ex_user = prompts['ex_user']
    ex_assistant = prompts['ex_assistant']
    context_prompt_user = prompts['context_prompt_user']
    prompt = base_prompt+ex_user+ex_assistant+context_prompt_user+text
    body = {
        "model":model_name,
        "input":prompt,
        "reasoning":{"effort":effort},
        "text":{"format":text_format},
        }
    bitem["body"] = body
    return bitem

def read_batch_response_contents(lines, model_type, perturbation_type):
    completions = []
    #Slightly complicated way of getting around some output format errors and still getting all response texts
    #Should work always, but may need to change in the future
    for line in lines:
        infor = json.loads(line)
        cust_id = infor['custom_id']
        tex = infor['response']['body']['output']
        temp = ""
        for t in tex:
            has_text = t.get('content', None)
            if has_text:
                if isinstance(has_text, list):
                    has_text = has_text[0]
                has_text = has_text['text']
                if isinstance(has_text, str):
                    if has_text[:9] == "{\"text\":\"":
                        temp+=has_text[9:-1]
                    else:
                        temp += has_text
                else:
                    try:
                        temp += has_text['text'].get('text', '')
                    except:
                        pprint(has_text)
        completions.append({"perturbation_type":perturbation_type, "model":model_type, "head_id":cust_id, "text":temp.replace("'''\\n", "").replace("'''", '')})
    return completions


def main(cmd_args):

    model_name = cmd_args[0]
    effort = cmd_args[1]
    custom_dataset_name = cmd_args[2]
    perturbation_layer = int(cmd_args[3])
    if len(cmd_args) > 4:
        existing_batch_job_id = cmd_args[4]

    #Can change implementation later
    perturbation_type = "clumsification"
    language = "english"

    response_format = type_to_text_format_param(EditedTexts)

    if perturbation_layer == 0:
        input_name = "original.jsonl"
        output_name = PERTURBATION_FOLDER+"1.jsonl"
    else:
        input_name = PERTURBATION_FOLDER+str(perturbation_layer)+".jsonl"
        output_name = PERTURBATION_FOLDER+str(perturbation_layer+1)+".jsonl"
    
    custom_ds_folder = BASE_FOLDER+custom_dataset_name+"/"

    #If no cache file exists
    if not os.path.exists(custom_ds_folder+str(perturbation_layer+1)+".jsonl"):
        batch_items = []
        with open(custom_ds_folder+input_name, 'r') as reader:
            for l in reader:
                og_text = json.loads(l)['text']
                og_id = json.loads(l)['head_id']
                batch_items.append(generate_batch_item(input_name, og_id, og_text, model_name, effort, response_format, language, perturbation_type))

        
        with open(custom_ds_folder+str(perturbation_layer+1)+".jsonl", 'w') as writer:
            for f in batch_items:
                writer.write(json.dumps(f)+'\n')

    #If we want to submit a batch job with the existing cache file (usually the remaining, uncompleted items)
    if not len(cmd_args)>4:
        batch_input_file = client.files.create(
            file=open(custom_ds_folder+str(perturbation_layer+1)+".jsonl", "rb"),
            purpose="batch"
        )

        #print(batch_input_file)
        print("Batch file created! Proceeding to send it to Open AI API")

        batch_input_file_id = batch_input_file.id
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": "batch processing job"
            }
        )

        batch_job = client.batches.retrieve(batch_job.id)
    #If we want to process some batch job that already completed
    else:
        batch_job = client.batches.retrieve(existing_batch_job_id)
    
    print("Initial batch job information:\n")
    print(batch_job)
    ss = ['validating', 'in_progress', 'finalizing', 'cancelling']
    print("\nWaiting for response...\n")
    while client.batches.retrieve(batch_job.id).status in ss:
        time.sleep(300)
    print("Got a response from the API:\n")
    batch_job = client.batches.retrieve(batch_job.id)
    print(batch_job)
    if batch_job.error_file_id:
        print('\n\n\n')
        #print(client.files.content(batch_job.error_file_id).content)
        #Get all successfully processed items in the batch
        if batch_job.output_file_id:
            result_file_id = batch_job.output_file_id
            file_response = client.files.content(result_file_id)

            completions = read_batch_response_contents(file_response.read().splitlines(), model_name, perturbation_type)
            
            result_file_name = custom_ds_folder+output_name

            with open(result_file_name, 'a') as file:
                for f in completions:
                    file.write(json.dumps(f)+'\n')
        #Create a new file that contains the rest of the batch items which failed
        compl_ids = [x['custom_id'] for x in completions]
        not_completed = []
        with open(custom_ds_folder+str(perturbation_layer+1)+".jsonl", 'r') as reader:
            for f in reader:
                if len(f) > 0:
                    t = json.loads(f.strip())
                    if t['custom_id'] not in compl_ids:
                        not_completed.append(t)

        with open(custom_ds_folder+str(perturbation_layer+1)+".jsonl", 'w') as writer:
            for f in not_completed:
                writer.write(json.dumps(f)+'\n')
        #Write the job_id to cache
        with open(custom_ds_folder+batch_job.id, 'w') as writer:
            writer.write('Failed')
        
        
            
    else:
        result_file_id = batch_job.output_file_id
        file_response = client.files.content(result_file_id)
        completions = read_batch_response_contents(file_response.read().splitlines(), model_name, perturbation_type)
        result_file_name = custom_ds_folder+output_name

        with open(result_file_name, 'a') as file:
            for f in completions:
                file.write(json.dumps(f)+'\n')
        os.remove(custom_ds_folder+str(perturbation_layer+1)+".jsonl")
if __name__ == "__main__":
    main(sys.argv[1:])
