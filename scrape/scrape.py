import requests
from bs4 import BeautifulSoup
import re
import json


def remove_html_tags(data):
  p = re.compile(r'<.*?>')
  return p.sub('', data)


def page_to_json(page_contents, fname):
    with open(fname, 'w', encoding='utf-8') as fp:
        json.dump(page_contents, fp, indent=4, ensure_ascii=False)


def get_page_contents(URL):
    '''Scrape the contents based on the h2 headers and structure they use for a web page.

    URL: string, absolute URL to the page

    returns
    ---
    page_contents: dict, with key=subject, value=list of all paragraphs
    '''

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    page_contents = {}

    #Extract top page warnings, if any
    warnings = soup.find_all(class_="frame frame-avisExclam frame-type-textmedia frame-layout-0")
    if warnings:
        page_contents['page alerts'] = [warning.get_text() for warning in warnings if warnings]

    # Look for subjects and split them by header
    subjects = soup.find_all(class_='frame frame-default frame-type-textmedia frame-layout-0')
    for sub in subjects:

        # Look for headers in each subject
        raw_title = None
        if sub.find('h2'):
            raw_title = sub.find('h2').contents[0]
        elif sub.find('h3'):
            raw_title = sub.find('h3').contents[0]

        if raw_title:
            title = ' '.join(raw_title.split())
            # Extract the text from a header block
            if sub.find(class_='ce-bodytext'):
                raw_sub_text = sub.find(class_='ce-bodytext').contents
                subject_text = []
                for text in raw_sub_text:
                    subject_text.append(text.get_text())

                if title not in ['Avis', 'Notice']: # This is on every page
                    page_contents[title]=subject_text

    return page_contents


def get_english_page(URL, base_url='https://www.quebec.ca'):
    '''Get the english page associated to the french page'''

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    # The english translation link is the first link in an object called listePiv
    en_href = soup.find(class_='listePiv').find_all('a')[0].get('href')
    en_URL = base_url + en_href

    return en_URL


def page_to_md(page_contents, page_url, response_filename, nlu_filename):
    section_title = page_url.rsplit('/')[-2]

    for idx, (question, answer) in enumerate(page_contents.items()):

        intent = "web_content/" + str(idx) + section_title
        with open(response_filename, "a") as f:
            f.write("## " + str(idx) + section_title)
            f.write("\n")
            f.write("* " + intent)
            f.write("\n")
            f.write("  - " + " ".join(answer))
            f.write("\n")
            f.write("\n")

        with open(nlu_filename, "a") as f:
            f.write("## intent: " + intent)
            f.write("\n")
            for a in answer:
                f.write("  - " + a)
                f.write("\n")
            f.write("\n")

def get_faq_contents(faq_URL):

    page = requests.get(faq_URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    faq_contents = {}
    all_panels = soup.find_all(class_='panel panel-default')

    # a panel contains both the question and the answer
    for panel in all_panels:
        question = panel.find(class_='accordion-toggle')
        answer = panel.find(class_='ce-bodytext')

        if question and answer:
            faq_contents[question.get_text()] = [answer.get_text()]

    return faq_contents


def get_mainpage_contents(mainpage_URL):

    page = requests.get(mainpage_URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    page_contents = {}

    warnings = soup.find_all(class_="alert alert-warning")
    if warnings:
        page_contents['page alerts'] = [warning.get_text() for warning in warnings if warnings]

    # Look for subjects and split them by subtopic
    topics = soup.find_all(class_='frame frame-default frame-type-textmedia frame-layout-0')

    for topic in topics:

        # Look for headers in each subject
        raw_title = None
        if topic.find('h2'):
            raw_title = topic.find('h2').contents[0]
        if raw_title:
            title = ' '.join(raw_title.split())
            link_names = [href.get_text() for href in topic.find_all('a')]
            page_contents[title] = link_names

    return page_contents


if __name__ == '__main__':

    # faq page is different in structure so is parsed separately below.
    french_URLS = [
        'https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/situation-coronavirus-quebec/',
        'https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/consignes-directives-contexte-covid-19/',
        'https://www.quebec.ca/sante/problemes-de-sante/a-z/informations-generales-sur-le-coronavirus/',
        'https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/stress-anxiete-et-deprime-associes-a-la-maladie-a-coronavirus-covid-19/',
        'https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/consignes-directives-contexte-covid-19/communautes-autochtones/',
    ]

    # This is to have the files compatible with the Raza framework
    # Note that content is being appended to the end of the .md files
    # They need to be deleted if already present when generating them
    # To generate the md files, switch convert_to_md to True
    convert_to_md = False
    responses_fr_fname = 'responses_fr.md'
    nlu_fr_fname = 'nlu_fr.md'
    responses_en_fname = 'responses_en.md'
    nlu_en_fname = 'nlu_en.md'

    # scrape the "regular pages" structures
    for count, fr_URL in enumerate(french_URLS):
        en_URL = get_english_page(fr_URL)
        page_contents_fr = get_page_contents(fr_URL)
        page_contents_en = get_page_contents(en_URL)

        page_to_json(page_contents_fr, str(count) + '_fr.json')
        page_to_json(page_contents_en, str(count) + '_en.json')


        if convert_to_md:
            page_to_md(page_contents_fr, fr_URL, responses_fr_fname, nlu_fr_fname)
            page_to_md(page_contents_en, en_URL, responses_en_fname, nlu_en_fname)


    # scrape the faq
    faq_URL_fr = 'https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/reponses-questions-coronavirus-covid19/'
    faq_URL_en = get_english_page(faq_URL_fr)
    faq_contents_fr = get_faq_contents(faq_URL_fr)
    faq_contents_en = get_faq_contents(faq_URL_en)
    page_to_json(faq_contents_fr, 'faq_fr.json')
    page_to_json(faq_contents_en, 'faq_en.json')

    # scrape the main page
    mainpage_URL_fr = 'https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/'
    mainpage_URL_en = get_english_page(mainpage_URL_fr)
    mainpage_contents_fr = get_mainpage_contents(mainpage_URL_fr)
    mainpage_contents_en = get_mainpage_contents(mainpage_URL_en)

    page_to_json(mainpage_contents_fr, 'mainpage_fr.json')
    page_to_json(mainpage_contents_en, 'mainpage_en.json')
