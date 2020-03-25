import json
import pandas as pd

def page_to_json(page_contents, fname):
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(page_contents, fp, indent=4, ensure_ascii=False)


def excel_to_page(excel_filename):
    xls = pd.ExcelFile(excel_filename)
    page_contents = {
        'document_URL': 'MSSS'
    }

    for page_number in range(1, 50):
        df = pd.read_excel(xls, page_number)

        if df.empty:
            continue

        answer = df.columns[2]
        questions = [df.columns[0]]

        if not df.empty:
            questions.extend(list(df[df.columns[0]].values))

        for question in questions:
            page_contents[question] = {
                'plaintext': [answer],
                'URL': 'MSSS_'+ str(page_number),
                'html': 'no html',
            }

    return page_contents

filename_en = './MSSS-Covid19-QnA-en_2020-03-20.xlsx'
page_contents_en = excel_to_page(filename_en)
page_to_json(page_contents_en, 'MSSS_faq_en.json')

filename_fr = './MSSS-Covid19-QnA-fr_2020-03-20.xlsx'
page_contents_fr = excel_to_page(filename_fr)
page_to_json(page_contents_fr, 'MSSS_faq_fr.json')
