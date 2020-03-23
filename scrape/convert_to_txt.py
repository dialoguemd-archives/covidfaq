#!/usr/bin/env python

import argparse
import csv
import glob
import json
import logging
import os

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-folder', help='folder with json input files.')
    parser.add_argument('--csv-folder', help='folder with csv input files.')
    parser.add_argument('--language', help='either fr or en', required=True)
    parser.add_argument('--output', help='will write the sentences to this file', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.output, 'w') as out_stream:
        if args.json_folder is not None:
            extract_sentences_from_json(args.json_folder, args.language, out_stream)
        else:
            logger.info('no json folder specified - ignoring json..')
        if args.csv_folder is not None:
            extract_sentences_from_csv(args.csv_folder, out_stream)
        else:
            logger.info('no csv folder specified - ignoring csv..')


def extract_sentences_from_csv(csv_folder, out_stream):
    for f in glob.glob(os.path.join(csv_folder, '*.csv')):
        logger.info('loading data from {}'.format(f))
        with open(f, 'r') as instream:
            reader = csv.reader(instream)
            for row in reader:
                out_stream.write(row[2].strip() + '\n')


def extract_sentences_from_json(json_folder, language, out_stream):
    for f in glob.glob(os.path.join(json_folder, '*{}.json'.format(language))):
        logger.info('loading data from {}'.format(f))
        with open(f, 'r') as instream:
            f_dict = json.load(instream)
            for k, v in f_dict.items():
                if k == 'document_URL':
                    continue
                sentences = v['plaintext']
                assert type(sentences) == list
                for sentence in sentences:
                    if sentence.strip() != '':
                        out_stream.write(sentence.strip() + '\n')


if __name__ == '__main__':
    main()
