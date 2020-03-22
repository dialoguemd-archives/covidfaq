import argparse
import csv
import glob
import json
import os

import torch
from transformers import *

logger = logging.getLogger(__name__)


def get_master_questions(folder, use_content):
    master_questions = []
    for f in glob.glob(os.path.join(folder, '*fr.json')):
        logger.info('loading master questions from {}'.format(f))
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


def encode_sentences(sentences, tokenizer, model):
    encoded_sentences = []
    related_input_sentences = []
    for sentence in sentences:
        encoded = encode(sentence, tokenizer, model)
        if encoded is not None:
            encoded_sentences.append(encoded)
            related_input_sentences.append(sentence)
    logger.info('encoded {} sentences from {} - skipped {}'.format(
        len(encoded_sentences), len(sentences), len(sentences) - len(encoded_sentences)
    ))
    return encoded_sentences, related_input_sentences


def encode(sentence, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
    if input_ids.shape[1] >= 512:
        logger.warning('skipping sentence because too long: {}'.format(sentence))
        return None  # too long
    with torch.no_grad():
        input_ids = input_ids[:, :512]  # flaubert limitation
        last_hidden_states = model(input_ids)[0][0]
        # return last_hidden_states[0]  # just the CLS.
        return torch.mean(last_hidden_states, 0)  # mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mq-folder', help='folder containing master question json files',
                        required=True)
    parser.add_argument('--uq-csv', help='csv file containing user questions',
                        required=True)
    parser.add_argument('--output', help='output file', required=True)
    parser.add_argument('--use-content', help='use also the content of the master question'
                        'paragraph for similarity', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    pretrained_weights = 'flaubert-large-cased'
    tokenizer_class = FlaubertTokenizer
    model_class = FlaubertModel
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    user_qs = get_user_questions(args.uq_csv)
    logger.info('encoding user questions')
    encoded_user_qs, related_user_qs = encode_sentences(user_qs, tokenizer, model)

    master_qs = get_master_questions(args.mq_folder, args.use_content)
    logger.info('encoding master questions')
    encoded_master_qs, related_master_qs = encode_sentences(master_qs, tokenizer, model)

    with open(args.output, 'w') as outstream:
        for i, user_q in enumerate(related_user_qs):
            outstream.write('\nuq: {}\n'.format(user_q))
            scores = []
            for j, master_q in enumerate(related_master_qs):
                dot = torch.dot(encoded_user_qs[i], encoded_master_qs[j])
                scores.append((dot, master_q))
            for dot, master_q in reversed(sorted(scores)):
                outstream.write('\tmq: {} => {}\n'.format(master_q, dot))


if __name__ == '__main__':
    main()
