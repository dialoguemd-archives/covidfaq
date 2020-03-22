import argparse
import csv
import glob
import json
import os

import torch
from transformers import *


def get_master_questions(folder, use_content):
    master_questions = []
    for f in glob.glob(os.path.join(folder, '*fr.json')):
        print(f)
        with open(f, 'r') as instream:
            f_dict = json.load(instream)
            for k, v in f_dict.items():
                if use_content:
                    if type(v) == list:  # fix for json format bug
                        to_add = k + ' ' + ' '.join(v)
                    else:
                        to_add = k + ' ' + v
                else:
                    to_add = k
                master_questions.append(to_add)
    return master_questions


def get_user_questions(file_path):
    result = []
    with open(file_path, 'r') as instream:
        reader = csv.reader(instream)
        for row in reader:
            result.append(row[2])
    return result


def encode(sentence, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0][0]
        # return last_hidden_states[0]  # just the CLS.
        return torch.mean(last_hidden_states, 0)  # mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mq-folder', help='folder containing master question json files',
                        required=True)
    parser.add_argument('--uq-csv', help='csv file containing user questions',
                        required=True)
    parser.add_argument('--use-content', help='use also the content of the master question'
                        'paragraph for similarity', action='store_true')
    args = parser.parse_args()

    pretrained_weights = 'flaubert-large-cased'
    tokenizer_class = FlaubertTokenizer
    model_class = FlaubertModel
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    user_qs = get_user_questions(args.uq_csv)
    encoded_user_qs = [encode(x, tokenizer, model) for x in user_qs]

    master_qs = get_master_questions(args.mq_folder, args.use_content)
    encoded_master_qs = [encode(x, tokenizer, model) for x in master_qs]

    for i, user_q in enumerate(user_qs):
        print('uq: {}'.format(user_q))
        scores = []
        for j, master_q in enumerate(master_qs):
            dot = torch.dot(encoded_user_qs[i], encoded_master_qs[j])
            scores.append((dot, master_q))
        for dot, master_q in reversed(sorted(scores)):
            print('\tmq: {} => {}'.format(master_q, dot))


if __name__ == '__main__':
        main()

