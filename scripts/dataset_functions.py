import json
import random
import os
from datasets import Dataset, Features, Value, Sequence

# Data should be in format:
# {col, col_id, text, under_col, under_col_id}
def format_datasets(dss:list[dict[str]]):
    # Dicts are in format: {ds_path, ds_name, under_ds_name}
    ds_items = []
    for x in dss:
        ds_path = x['ds_path']
        ds_name = x['ds_name']
        ds_under = x['under_ds_name']
        with open(ds_path, 'r', encoding='utf-8') as reader:
            for i,l in enumerate(reader):
                if len(l)>0:
                    line = json.loads(l.strip())
                    if ds_under:
                        ds_items.append({'collection':ds_name, 'collection_id':ds_name+"_"+str(i), 'text':line['text'], 'under_collection':ds_under, 'under_collection_id':line['custom_id']})
                    else:
                        ds_items.append({'collection':ds_name, 'collection_id':ds_name+"_"+str(i), 'text':line['text'], 'under_collection':None, 'under_collection_id':None})



    return ds_items

def init_og_dataset(source_txt_path: str, new_ds_name: str, overwrite: bool = False):
    #If a txt-file with one text per line, then create a jsonl-file so that it fits with the rest of the code
    #Also creates the source folder for the specified dataset if necessary
    if not os.path.exists("data/custom_datasets/"+new_ds_name):
        os.mkdir("data/custom_datasets/"+new_ds_name)
        os.mkdir("data/custom_datasets/"+new_ds_name+"/perturbed_layers")
    if not os.path.exists("data/custom_datasets/"+new_ds_name+"/original.jsonl") or overwrite:
        with open("data/custom_datasets/"+new_ds_name+"/original.jsonl", 'w', encoding='utf-8') as writer:
            with open(source_txt_path, 'r', encoding='utf-8') as reader:
                i = 0
                for line in reader:
                    writer.write(json.dumps({'custom_id':str(i),'text':line.replace('\n', '')})+'\n')
                    i+=1
    return "data/custom_datasets/"+new_ds_name+"/original.jsonl"


def read_ds(ds_path: str):
    #The most simple of helper functions
    returnable = []
    with open(ds_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                if len(line) > 1:
                    returnable.append(json.loads(line.strip()))
    return returnable

def format_custom_dataset(custom_dataset_name: str):
    """
    Function for getting the existing files inside custom dataset folders into usable form
    """
    work_path = "data/custom_datasets/"+custom_dataset_name
    original_texts = read_ds(work_path+"/original.jsonl")
    #Using a lsit of tuples to make shuffling easier
    #After all, it's trivial to then later on get two lists
    id_dict = {i:{'id':i, 'text_label_pairs':[(x['text'],0)]} for i,x in enumerate(original_texts)}
    #Move onto adding perturbed layers one at a time
    for file_name in os.listdir(work_path+"/perturbed_layers"):
        contents = read_ds(work_path+"/perturbed_layers/"+file_name)
        layer = int(file_name.replace('.jsonl', ''))
        #Go through each row in the perturbed layer-file, and add to the same object with its original version
        for x in contents:
            head_id = x['head_id']
            if isinstance(head_id, str):
                head_id = int(head_id)
            temp = id_dict[head_id]
            temp['text_label_pairs'] += [(x['text'],layer)]
            id_dict[head_id] = temp
    #The extra id from the dict sturcture has served its purpose, so return only the contents (a list of dicts)
    return list(id_dict.values())


def shuffle_and_transform_formatted_dataset(formatted_dataset:list[dict], seed: int=None):
    #First apply shuffling to each internal 'text_label_pairs' list
    if seed:
        random.seed(seed)
    for x in formatted_dataset:
        tl_list = x.pop('text_label_pairs', None)
        random.shuffle(tl_list)
        #After shuffling, separate texts and labels to their own lists
        x['texts'] = [y[0] for y in tl_list]
        x['labels'] = [y[1] for y in tl_list]

    return Dataset.from_list(formatted_dataset)
        


            


def sample_reference_corpus(dict_list, reference_name, reference_size):
    """
    Sample dictionaries from a list where the 'collection' field equals reference_name,
    and remove both sampled dictionaries and related dictionaries based on collection_id.
    
    Args:
        dict_list: List of dictionaries with format {'collection', 'collection_id', 'text', 
                  'under_collection', 'under_collection_id'}.
        reference_name: The value to match with the 'collection' field.
        reference_size: Number of dictionaries to sample.
    
    Returns:
        A tuple of (sampled_list, remaining_list) where:
        - sampled_list: List of sampled dictionaries from the specified collection
        - remaining_list: Original list with sampled dictionaries and related dictionaries removed
    """
    # Filter the list to find all dictionaries where 'collection' equals reference_name
    candidates = [d for d in dict_list if d.get('collection') == reference_name]
    
    # Ensure we don't try to sample more than what's available
    sample_size = min(reference_size, len(candidates))
    
    # Randomly sample from the candidates
    sampled = random.sample(candidates, sample_size)
    
    # Get collection_ids of all sampled items
    sampled_collection_ids = {d.get('collection_id') for d in sampled}
    
    # Create a new list that excludes:
    # 1. The sampled dictionaries
    # 2. Dictionaries where 'under_collection_id' equals 'collection_id' of any sampled dictionary
    remaining = [d for d in dict_list if d not in sampled and 
                 d.get('under_collection_id') not in sampled_collection_ids]
    
    return sampled, remaining

