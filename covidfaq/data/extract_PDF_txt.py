import json

import pandas as pd


def page_to_json(page_contents, fname):
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(page_contents, fp, indent=4, ensure_ascii=False)


def txt_to_page(txt_filename):
    with open(txt_filename) as data:
        raw_text = data.read()
        lines = raw_text.split('\n')

    page_contents = {"document_URL": "PDF"}

    question = []
    answer = []
    for line in lines:

        if not question:
            question = line

        elif line != '':
            answer.append(line)

        elif line == '':
            assert type(answer) == list
            page_contents[question] = {
                'plaintext': answer,
                'URL': 'PDF',
                'html': 'no html',
            }
            question = []
            answer = []
            continue

    return page_contents


filename_en = "./19-210-30A_Guide-auto-soins_anglais.txt"
page_contents_en = txt_to_page(filename_en)
page_to_json(page_contents_en, "PDF_faq_en.json")

filename_fr = "./19-210-30FA_Guide-autosoins_francais.txt"
page_contents_fr = txt_to_page(filename_fr)
page_to_json(page_contents_fr, "PDF_faq_fr.json")
