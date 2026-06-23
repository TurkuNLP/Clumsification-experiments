#Imports
import os
import argparse
import json
from tqdm.auto import tqdm

#Arparsing
def parse_args():
    parser = argparse.ArgumentParser(
            description="Transform CONLLU documents downloaded from UD docs into a format that will be used to create mutlilingual evals."
        )
    
    parser.add_argument(
        "--base-folder",
        type=str,
        required=True,
        help="The benchmark folder that stores the ud data docs"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite an existing original doc data file"
    )

    return parser.parse_args()

#Functions for taking a raw conllu and turning it into a jsonl with full documents
#Placeholder function, migrating rest of the TDT code later
def parse_conllu_fin(folder_path: str):
    texts = []
    metadata = []
    genres = []
    for fn in os.listdir(folder_path):
        if fn.endswith(".conllu"):
            with open(folder_path+fn, 'r') as reader:
                for l in reader:
                    if l.find("# sent_id = ") != -1:
                        metadata.append(l[12:-1])
                        continue
                    if l.find(" text = ") != -1:
                        texts.append(l[8:-1])
    return texts, metadata, genres

#Helper functions for English
#Function for parsing conllus based on the GUM corpus
def parse_conllu_gum(file_path: str):

    genre_map = {
        'whow': 'wikihow',
        'voyage': 'travel guide',
        'bio': 'biography'
    }

    texts = []
    titles = []
    t_types = []
    genres = []
    doc_ids = []
    sent_amounts = []

    with open(file_path, 'r', encoding='utf-8') as reader:
        md = ""
        g = ""
        title = ""
        t_type = ""
        text_parts = []
        sent_amount = 0
        in_doc = False

        def flush_current_doc():
            if not in_doc:
                return

            # Avoid appending completely empty docs
            if title or text_parts:
                texts.append(" ".join(text_parts).strip())
                genres.append(g)
                titles.append(title)
                doc_ids.append(md)

                if t_type:
                    t_types.append(t_type)
                else:
                    t_types.append("title")

                sent_amounts.append(sent_amount)

        for l in reader:
            line = l.rstrip("\n")

            if line.startswith("# newdoc id = "):
                # Flush previous document before starting a new one
                flush_current_doc()

                md = line[len("# newdoc id = "):].strip()
                g = ""
                title = ""
                t_type = ""
                text_parts = []
                sent_amount = 0
                in_doc = True

                continue

            elif line.startswith("# meta::genre = "):
                g = line[len("# meta::genre = "):].strip()
                g = genre_map.get(g, g)
                continue

            elif line.startswith("# meta::title = "):
                title = line[len("# meta::title = "):].strip()
                continue

            elif line.startswith("# text = "):
                sent_text = line[len("# text = "):].strip()

                # If the title also appears as the first text line,
                # treat it as a first-line title/header and do not include it
                # in the main document text.
                if title and sent_text == title and not text_parts:
                    t_type = "first_line"
                else:
                    text_parts.append(sent_text)
                    sent_amount += 1

                continue

        # Flush the final document in the file
        flush_current_doc()

    return texts, titles, t_types, genres, doc_ids, sent_amounts

#Function for parsing conllus based on the EWT corpus
def parse_conllu_ewt(file_path: str):

    genre_map = {
        "weblog": "web blog",
        "answers": "question answering",
        "reviews": "review",
        "email": "email",
        "newsgroups": "newsgroup",
    }

    texts = []
    titles = []
    t_types = []
    genres = []
    doc_ids = []
    sent_amounts = []

    def infer_genre_from_doc_id(doc_id: str) -> str:
        """
        EWT doc ids look like:
        weblog-blogspot.com_nominations_...
        answers-201111080928...
        reviews-...
        
        The part before the first '-' is the coarse genre.
        """
        raw_genre = doc_id.split("-", 1)[0]
        return genre_map.get(raw_genre, raw_genre)

    with open(file_path, "r", encoding="utf-8") as reader:
        md = ""
        g = ""
        first_line = ""
        text_parts = []
        sent_amount = 0
        in_doc = False

        def flush_current_doc():
            if not in_doc:
                return

            if first_line or text_parts:
                texts.append(" ".join(text_parts).strip())
                titles.append(first_line)
                t_types.append("first_line")
                genres.append(g)
                doc_ids.append(md)
                sent_amounts.append(sent_amount)

        for l in reader:
            line = l.rstrip("\n")

            if line.startswith("# newdoc id = "):
                # Save previous document before starting a new one
                flush_current_doc()

                md = line[len("# newdoc id = "):].strip()
                g = infer_genre_from_doc_id(md)

                first_line = ""
                text_parts = []
                sent_amount = 0
                in_doc = True

                continue

            elif line.startswith("# text = "):
                sent_text = line[len("# text = "):].strip()

                if not first_line:
                    # EWT has no title metadata, so store the first sentence separately
                    first_line = sent_text
                else:
                    text_parts.append(sent_text)
                    sent_amount += 1

                continue

        # Important: flush the last document in the file
        flush_current_doc()

    return texts, titles, t_types, genres, doc_ids, sent_amounts

def parse_conllu_en(folder_path: str):

    texts = []
    titles = []
    t_types = []
    genres = []
    doc_ids = []
    sent_amounts = []

    for fn in os.listdir(folder_path):
        if fn.endswith(".conllu"):
            file_path = os.path.join(folder_path, fn)

            if "gum" in fn.lower():
                t_texts, t_titles, t_t_types, t_genres, t_doc_ids, t_sent_amounts = parse_conllu_gum(file_path)

                texts += t_texts
                titles += t_titles
                t_types += t_t_types
                genres += t_genres
                doc_ids += t_doc_ids
                sent_amounts += t_sent_amounts

            elif "ewt" in fn.lower():
                t_texts, t_titles, t_t_types, t_genres, t_doc_ids, t_sent_amounts = parse_conllu_ewt(file_path)

                texts += t_texts
                titles += t_titles
                t_types += t_t_types
                genres += t_genres
                doc_ids += t_doc_ids
                sent_amounts += t_sent_amounts

    return texts, doc_ids, titles, t_types, genres, sent_amounts


#Moved to using the Parallel UD sentences as "generation seeds" for ranking models of the same family
#So we now only need one function (the documents have the exact same format!)

#Writer

def write_jsonl(data: list, base_folder: str, lan: str, overwrite:bool=False):

    f_path = base_folder+lan+"/ud_data.jsonl"

    if not os.path.exists(f_path) or overwrite:
        with open(f_path, 'w') as writer:
            first = True
            for l in data:
                if first:
                    writer.write(json.dumps(l))
                    first = False
                else:
                    writer.write('\n'+json.dumps(l))

    

#Main function

def main():

    args = parse_args()

    base_folder = args.base_folder

    g_map = {
        'n' : 'news',
        'w' : 'wiki'
    }

    for fol in tqdm(os.listdir(base_folder), desc="Parsing PUD conllus..."):
        titles = []
        genres = []
        doc_ids = []
        with open(base_folder+fol+"/"+fol+"_pud-ud-test.conllu") as reader:
            for l in reader:
                line = l.rstrip("\n")
                if line.startswith("# sent_id = "):
                    t = line.replace("# sent_id = ", "")
                    doc_ids.append(t)
                    genres.append(g_map[t[0]])
                elif line.startswith("# text = "):
                    sent_text = line[len("# text = "):].strip()
                    titles.append(sent_text)
        full_docs_list = []
        for i in range(len(titles)):
            full_docs_list.append({
                'id':doc_ids[i], 
                'register':genres[i], 
                'prompt_sentence':titles[i], 
                'text':"", 
                'text_sent_amount':"12-15"
            })

        write_jsonl(full_docs_list, base_folder, fol, args.overwrite)


if __name__ == "__main__":
    main()