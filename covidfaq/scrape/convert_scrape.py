import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)


def collapse_jsons(json_files):
    collapsed = {}
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as fh:
            faq = json.load(fh)
            for k, v in faq.items():
                if k != "document_URL":
                    collapsed[k] = v
    return collapsed


def json_to_passages(collapsed_json, passage_id_start_idx=0):

    passages = []
    passage_id = passage_id_start_idx
    for entry in collapsed_json.values():
        passage = {
            "passage_id": passage_id,
            "source": entry['source'],
            "uri": entry['url'],
            "reference_type": entry['type'],
            "reference": {
                "page_title": "".join(entry['nested_title']).strip(),
                "section_headers": [entry['title']],
                "section_content": "".join(entry['plaintext']).strip(),
                "selected_span": None,
            }
        }
        passage_id += 1
        passages.append(passage)

    logger.info('generated  {} passages from the scrape provided'.format(len(passages)))

    return passages


def dump_passages(passages, fname):
    with open(fname, "w", encoding='utf-8') as f:
        json.dump({'passages': passages}, f, indent=6, ensure_ascii=False)


def get_scraped_json_filenames(scrapes_path, source, lang, is_faq=True):
    matches = [source, lang, '.json']
    if is_faq:
        matches += ['faq']
        return [scrapes_path + f for f in os.listdir(scrapes_path) if all(match in f for match in matches)]
    else:
        return [scrapes_path + f for f in os.listdir(scrapes_path) if all(match in f for match in matches) and 'faq' not in f]


def scrapes_to_passages(scrapes_path, source, lang, is_faq):
    '''Writes the scrapes to the proper format in output_file'''
    json_filenames = get_scraped_json_filenames(scrapes_path, source, lang, is_faq)
    collapsed_json = collapse_jsons(json_filenames)
    passages = json_to_passages(collapsed_json)
    return passages



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input json (list of files to convert, can use a regex, e.g. quebec-en-faq*.json ", required=True, nargs='+')
    parser.add_argument("--output-passages", help="output file in bert_reranker format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load json scrapes and collapse them to a single json
    collapsed_json = collapse_jsons(args.input)
    logger.info('collapsed {} files into a single dict with {} elements'.format(
        len(args.input), len(collapsed_json)
    ))

    # generate passages from the scrape to the appropriate format
    passages = json_to_passages(collapsed_json)
    if args.output_passages is not None:
        dump_passages(passages, args.output_passages)

if __name__ == "__main__":
    main()
