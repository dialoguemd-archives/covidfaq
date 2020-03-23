#!/usr/bin/env python

import argparse
import glob
import json
import logging
import os

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', help='folder with input files (default ./)',
                        default='./')
    parser.add_argument('--language', help='either fr or en', required=True)
    parser.add_argument('--output', help='will write the sentences to this file', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.output, 'w') as out_stream:
        extract_sentences(args, out_stream)


def extract_sentences(args, out_stream):
    for f in glob.glob(os.path.join(args.input_folder, '*{}.json'.format(args.language))):
        logger.info('loading data from {}'.format(f))
        with open(f, 'r') as instream:
            f_dict = json.load(instream)
            for k, v in f_dict.items():
                if k == 'document_URL':
                    continue
                sentences = v['plaintext']
                if type(sentences) is not list:
                    # possible bug in generating the json
                    sentences = [sentences]
                for sentence in sentences:
                    if sentence.strip() != '':
                        out_stream.write(sentence.strip() + '\n')


if __name__ == '__main__':
    main()
