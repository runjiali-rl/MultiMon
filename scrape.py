import openai
import re
import argparse
from tqdm import tqdm
import os
import json
import numpy as np
from datetime import datetime
import torch
from collections import defaultdict
import torch.nn.functional as F
import torch
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv 
from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer
import argparse
import urllib.request
import zipfile
import pickle

parser = argparse.ArgumentParser(description="Scraping from corpus data with steering (optional)")
    
parser.add_argument("--steer", type=str, default = "", help="Steering direction for the self-driving car")
parser.add_argument("--corpus_data", type=str, default = "MS-COCO", help="Corpus data to scrape")
parser.add_argument("--num_output", type=int, default = 150, help="Number of entries we want")
parser.add_argument("--api_key", type=str, default = None, help="API Key for openAI account")
parser.add_argument("--do_steer", action='store_true', default=False, help="Whether to steer the scraping")
parser.add_argument("--fp16", action='store_true', default=False, help="Whether to use fp16")


args = parser.parse_args()
unique_rows = set() # create a set to store unique rows

# Erik, your api_key
if args.do_steer:
    assert args.api_key is not None, "You need to provide an API key to steer the scraping"
# openai.api_key = args.api_key
client = openai.OpenAI(api_key=args.api_key)
# Define a function to query the OpenAI API and evaluate the answer
def get_yes_no_answer(question):
    prompt = f'Please respond with either "yes" or "no" to the following: {question}'
    message = [
        {'role': 'user', 'content': prompt},
    ]

    response = client.chat.completions.create(
        messages = message,
        model = "gpt-3.5-turbo"
    )

    answer = response.choices[0].message.content.strip()
    yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

    if yes_no_regex.match(answer):
        return answer.lower()
    else:
        return "Could not determine yes or no."


def load_bert_model():

    bert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    return bert_model

def download_and_extract_dataset_SNLI(url, data_dir):
    file_path = os.path.join(data_dir, 'snli_1.0.zip')

    # Check if the snli_1.0.zip is already downloaded
    if not os.path.exists(file_path):
        print("Downloading the SNLI dataset...")
        urllib.request.urlretrieve(url, file_path)

    # Check if the SNLI dataset is already extracted
    if not os.path.exists(os.path.join(data_dir, 'snli_1.0')):
        print("Extracting the SNLI dataset...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(file_path)


def load_snli():


    data_dir = '.'
    annotations_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"  # URL to download the SNLI dataset

    download_and_extract_dataset_SNLI(annotations_url, data_dir)

    # Define the path to the dataset file
    snli_train_file = "snli_1.0/snli_1.0_train.jsonl"
    snli_dev_file = "snli_1.0/snli_1.0_dev.jsonl"
    snli_test_file = "snli_1.0/snli_1.0_test.jsonl"

    # Read the dataset files using pandas
    train_data = pd.read_json(snli_train_file, lines=True)
    dev_data = pd.read_json(snli_dev_file, lines=True)
    test_data = pd.read_json(snli_test_file, lines=True)

    # Remove rows with '-' label (no label assigned)
    train_data = train_data[train_data['gold_label'] != '-']
    dev_data = dev_data[dev_data['gold_label'] != '-']
    test_data = test_data[test_data['gold_label'] != '-']
    # Collect all premises in the datasets
    train_premises = train_data['sentence1'].tolist()
    dev_premises = dev_data['sentence1'].tolist()
    test_premises = test_data['sentence1'].tolist()

    # Combine the premises from all splits
    all_premises = train_premises + dev_premises + test_premises

    # Remove duplicates if needed
    unique_premises = list(set(all_premises))

    return unique_premises


def load_captions(annotations_path):
    # Initialize COCO API
    coco = COCO(annotations_path)

    # Get all image IDs
    img_ids = coco.getImgIds()

    # Loop through all image IDs and get their captions
    all_captions = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_captions = [ann['caption'].lower() for ann in anns]
        all_captions.extend(img_captions)

    return all_captions



def download_and_extract_dataset_COCO(url, data_dir):
    os.makedirs(data_dir, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(data_dir, 'annotations.zip')

    # Check if the annotations.zip is already downloaded
    if not os.path.exists(file_path):
        print("Downloading the annotations...")
        urllib.request.urlretrieve(url, file_path)

    # Check if the annotations are already extracted
    if not os.path.exists(os.path.join(data_dir, 'annotations')):
        print("Extracting the annotations...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(file_path)


def load_panda(file_path):
    data = pd.read_csv(file_path)
    all_captions = []
    for caption_list in data['caption'].tolist():
        caption_list = caption_list.split(',')
        caption_list = [caption.replace("'", "") for caption in caption_list]
        caption_list = [caption.replace("[", "") for caption in caption_list]
        caption_list = [caption.replace("]", "") for caption in caption_list]
        caption_list = [caption.replace('"', "") for caption in caption_list]
        caption_list = [caption.strip() for caption in caption_list]
        all_captions.extend(caption_list)
    return all_captions


def load_coco():
    # Set the paths to the dataset and annotations files
    data_dir = 'coco_annotation'
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    download_and_extract_dataset_COCO(annotations_url, data_dir)

    annotations_train_path = os.path.join(data_dir, 'annotations', 'captions_train2017.json')
    annotations_val_path = os.path.join(data_dir, 'annotations', 'captions_val2017.json')

    # Load captions for both training and validation sets
    all_captions_train = load_captions(annotations_train_path)
    all_captions_val = load_captions(annotations_val_path)

    # Combine both lists of captions
    all_captions = all_captions_train + all_captions_val

    print(f"Total number of captions (train): {len(all_captions_train)}")
    print(f"Total number of captions (val): {len(all_captions_val)}")
    print(f"Total number of captions (train + val): {len(all_captions)}")

    return all_captions

def load_clip_model():

    # Load the pre-trained CLIP model
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    model = model.cuda()

    # Load the corresponding tokenizer
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    return model, tokenizer, processor

def write_unique_rows(row, writer):
    """
    Write unique rows to a CSV file, ignoring any rows that have already been written.
    """

    # Define a key as a tuple of the values in row[2] and row[3]
    key1 = (row[2], row[3])
    key2 = (row[0], row[1])
    key3 = (row[1], row[0])



    # Check if the key is already in the set, and write the row if it is not
    if (key1 not in unique_rows) and (key2 not in unique_rows) and (key3 not in unique_rows):
        unique_rows.add(key1)
        unique_rows.add(key2)
        unique_rows.add(key3)
        
        writer.writerow(row)

        return True

    return False
     

def scrape(clip_model, tokenizer, bert_model, premises, similarity_threshold = 0.9, max_embedding_length = 4e6):
    save_path = f'similar_from_{args.corpus_data}_do_steer_{args.do_steer}.pkl'
    num_premises = int(min(max_embedding_length, len(premises)))

    if os.path.exists(save_path):
        similar_pairs = pickle.load(open(save_path, 'rb'))
        del bert_model
        del clip_model
        del tokenizer
    else:
        batch_size = 512

        # Compute the embeddings for each batch of premises
        bert_text_embeds_prompts_list = []
        for i in tqdm(range(0, num_premises, batch_size)):

            premises_batch = premises[i:i+batch_size]
            with torch.no_grad():
                text_embeds_prompts_batch = bert_model.encode(premises_batch)

            text_embeds_prompts_batch = torch.from_numpy(text_embeds_prompts_batch)
            text_embeds_prompts_batch = text_embeds_prompts_batch.cuda()
            text_embeds_prompts_batch = F.normalize(text_embeds_prompts_batch, dim=1)

            bert_text_embeds_prompts_list.append(text_embeds_prompts_batch)

        # Concatenate the embeddings for all batches
  
        bert_text_embeds_prompts = torch.cat(bert_text_embeds_prompts_list, dim=0)
        bert_text_embeds_prompts = bert_text_embeds_prompts[:num_premises, :]
        del bert_text_embeds_prompts_list

        # split the premises into batches
        premises_batches = [premises[i:i+batch_size] for i in range(0, num_premises, batch_size)]

        # compute the embeddings for each batch of premises
        text_embeds_prompts = torch.zeros(num_premises, 768)\
            .to(bert_text_embeds_prompts.dtype).to(bert_text_embeds_prompts.device)
        for i, premises_batch in enumerate(tqdm(premises_batches)):
            start_idx = i * batch_size
            tok = tokenizer(premises_batch, return_tensors="pt", padding=True, truncation=True)

            for key in tok.keys():
                tok[key] = tok[key].cuda()
            with torch.no_grad():
                text_outputs = clip_model.text_model(**tok)
            text_embeds = text_outputs[1]
            text_embeds = clip_model.text_projection(text_embeds)
            text_embeds_prompt = F.normalize(text_embeds, dim=1)
            end_idx = min(start_idx + batch_size, num_premises)
            text_embeds_prompts[start_idx:end_idx, :] = text_embeds_prompt[:end_idx-start_idx, :]

        # Initialize an empty list to store similar pairs
        similar_pairs = []
        assert len(text_embeds_prompts) == len(bert_text_embeds_prompts),\
            "The number of premises and embeddings do not match"

        #  Move the text embeddings to the GPU
        # text_embeds_prompts = text_embeds_prompts.to(torch.float8)
        # bert_text_embeds_prompts = bert_text_embeds_prompts.to(torch.float8)

        # Iterate over batches of embeddings
        for i in tqdm(range(0, len(premises), batch_size)):
            batch_premises = premises[i:i+batch_size]
            batch_text_embeds_prompts = text_embeds_prompts[i:i+batch_size]
            bert_batch_text_embeds_prompts = bert_text_embeds_prompts[i:i+batch_size]

            # Compute the dot product between each pair of embeddings in the batch
            similarity_matrix = torch.matmul(batch_text_embeds_prompts,
                                             text_embeds_prompts.t())
            bert_similarity_matrix = torch.matmul(bert_batch_text_embeds_prompts,
                                                  bert_text_embeds_prompts.t())

            mask = (similarity_matrix > similarity_threshold) &\
                  (abs(similarity_matrix - bert_similarity_matrix) > 0.2)

            # Find the indices of the matching pairs
            j_indices, k_indices = torch.nonzero(mask.float(), as_tuple=True)

            # Collect the matching pairs and their similarity scores

            for j, k in zip(j_indices.tolist(), k_indices.tolist()):
                similarity_score = similarity_matrix[j, k].item()
                bert_similarity_score = bert_similarity_matrix[j, k].item()
                similar_pairs.append((batch_premises[j],
                                      premises[k],
                                      similarity_score,
                                      bert_similarity_score,
                                      similarity_score-bert_similarity_score))

        pickle.dump(similar_pairs, open(save_path, 'wb'))
        del text_embeds_prompts
        del bert_text_embeds_prompts
        del bert_model
        del clip_model
        del tokenizer
    # Write similar pairs to a CSV file
    file_path = f'similar_from_{args.corpus_data}_top{args.num_output}_do_steer_{args.do_steer}.csv'
    with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sample 1', 'Sample 2', 'CLIP Similarity', 'BERT Similarity', 'Difference'])
        negative_keywords = ["there is no", "unable", "does not", "do not", "am not", "no image", "no picture"]
        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        num_written = 1

        for pair in tqdm(similar_pairs):
            # Check if none of the negative keywords are present in the row
            if not any(keyword in field for field in pair[:2] for keyword in negative_keywords):
                # Ask your yes-no question
                prompt1, prompt2 = pair[0], pair[1]
                if args.do_steer:
                    question = f"""
                    Does the difference between two described scenarios, referred to\
                          as '{prompt1}' and '{prompt2}', suggest a contrasting type of\
                          movement or direction? Are these differences indicative of opposite actions, \
                          such as one scenario showing an entrance while the other shows an exit, or one depicting an ascent while \
                          the other a descent? Consider the following examples to elaborate on this inquiry:

                    Entering vs. Exiting:
                    Prompt1: "A person walking towards a building," suggests an action of entering.
                    Prompt2: "A person walking away from a building," indicates the opposite action, exiting.

                    Ascending vs. Descending:
                    Prompt1: "A bird flying upwards towards the sky," illustrates ascending motion.
                    Prompt2: "A bird diving down towards the ground," represents descending motion.

                    Through these examples, can we determine if the variations between \
                        '{prompt1}' and '{prompt2}' in each case distinctly signify a change \
                        in the direction or type of motion, highlighting contrasting actions \
                        like entering versus exiting, or ascending versus descending? Please respond with either 'yes' or 'no.'"""
                    answer = get_yes_no_answer(question)
                    if answer == "yes":
                        # Write the unique row to the output file
                        is_unique = write_unique_rows(pair, csv_writer)
                        if is_unique:
                            num_written += 1

                        if num_written == args.num_output:
                            print("I finished!")
                            exit()
                else:
                    is_unique = write_unique_rows(pair, csv_writer)
                    if is_unique:
                        num_written += 1

                    if num_written == args.num_output:
                        print("I finished!")
                        exit()




if __name__ == '__main__':
    model, tokenizer, processor = load_clip_model()
    bert_model = load_bert_model()

    if args.corpus_data == "SNLI":
        unique_premise = load_snli()
    elif args.corpus_data == "MS-COCO":
        unique_premise = load_coco()
    elif args.corpus_data == "Panda10M":
        unique_premise = load_panda('/homes/55/runjia/scratch/panda70m/panda70m_training_10m.csv')
    elif args.corpus_data == "Panda70M":
        unique_premise = load_panda('/homes/55/runjia/scratch/panda70m/panda70m_training_full.csv')
    else:
        raise ValueError("Invalid corpus data")
    
    if args.fp16:
        model = model.half()
        bert_model = bert_model.half()




    scrape(model, tokenizer, bert_model, unique_premise, max_embedding_length=1.5e6)
