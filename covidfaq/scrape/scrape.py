import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from unicodedata import normalize
from urllib.parse import urljoin

import bs4
import requests
import structlog
import yaml
from bs4 import BeautifulSoup, NavigableString
from selenium import webdriver
from yaml import load

from covidfaq.scrape.convert_scrape import scrapes_to_passages, dump_passages

log = structlog.get_logger(__name__)


def remove_html_tags(data):
    p = re.compile(r"<.*?>")
    return p.sub("", data)


def soup_to_html(filename, soup):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(soup))


def clean_page_contents(page_contents):
    """Remove weird formatting from html like nbsp"""
    for k, v in page_contents.items():
        if k == "document_URL":
            continue
        k = normalize("NFKD", k)
        v["plaintext"] = [normalize("NFKD", s) for s in v["plaintext"]]
        v["nested_title"] = [normalize("NFKD", s) for s in v["nested_title"]]
        v["title"] = normalize("NFKD", v["title"])
    return page_contents


def page_to_json(page_contents, fname):
    page_contents = clean_page_contents(page_contents)
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(page_contents, fp, indent=4, ensure_ascii=False)


def page_to_md(page_contents, page_url, response_filename, nlu_filename):
    section_title = page_url.rsplit("/")[-2]

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


def rule_nesting(soup, info, url, rule):
    exclude = rule["exclude"]
    results = []

    # Keep track of header levels (h1 - h4)
    nested_titles = ["", "", "", ""]
    prev_nest_level = 0
    # Get the main header of the page if it exists
    main_title = soup.select(".d-block")
    if main_title:
        if main_title[0].name == "h1":
            nested_titles[0] = main_title[0].get_text()

    subjects = soup.select(rule["parent"])
    for sub in subjects:
        title = sub.select_one(rule["title"]) if rule["title"] else True
        body = sub.select_one(rule["body"]) if rule["body"] else sub
        if not title:
            continue

        if title is True:
            raw_title = ""
        else:
            raw_title = title.get_text().strip()

            if title.name in "h1, h2, h3, h4":
                nest_level = int(title.name[1]) - 1
                if nest_level >= prev_nest_level:
                    nested_titles[nest_level] = raw_title
                else:
                    nested_titles[nest_level] = raw_title
                    for n in range(nest_level + 1, len(nested_titles)):
                        nested_titles[n] = ""
                prev_nest_level = nest_level

        if exclude["titles"]:
            skip_entry = False
            for title in exclude["titles"]:
                if re.search(title, raw_title):
                    skip_entry = True
            if skip_entry:
                continue

        if not body:
            continue

        # Exclusions
        if exclude["selector"] and sub.select_one(exclude["selector"]):
            continue

        raw_body = body.get_text().strip()
        if exclude["body"] and re.search(exclude["body"], raw_body):
            continue

        entry = {
            "nested_title": nested_titles.copy(),
            "title": raw_title,
            "plaintext": [
                elem.get_text() if isinstance(elem, bs4.Tag) else str(elem)
                for elem in body
            ],
            "html": body.prettify(),
            "url": url,
        }
        entry.update(info)
        entry.update(rule["info"])
        results.append(entry)

    return results


def rule_headers(soup, info, url, rule):

    results = []
    exclude = rule["exclude"]

    if rule["header_type"] == "goa":
        headers = soup.select(rule["headers"][0])
    elif rule["header_type"] == "markdown":
        headers = soup.find_all(rule["headers"])

    for header in headers:
        nextNode = header
        header_str = str(nextNode.text)
        if exclude["titles"]:
            skip_header = False
            for title in exclude["titles"]:
                if re.search(title, header_str):
                    skip_header = True
            if skip_header:
                continue
        entry = {
            "nested_title": [],  # TODO: implement
            "title": header_str,
            "plaintext": [],
            "url": url,
            "html": "",
            "time": info["time"],
            "language": info["language"],
        }
        entry.update(info)
        entry.update(rule["info"])
        while True:
            nextNode = nextNode.nextSibling
            if nextNode is None:
                break
            elif nextNode.name in rule["headers"]:
                break
            elif isinstance(nextNode, NavigableString):
                entry["plaintext"].append(str(nextNode))
                entry["html"] += str(nextNode)
                continue
            if nextNode.text:
                if nextNode.text != "":  # add and isinstance(nextNode, Tag)?
                    entry["plaintext"].append(str(nextNode.text))
                    entry["html"] += str(nextNode)

        results.append(entry)

    return results


def rule_sibling(soup, info, url, rule):
    subjects = soup.select(rule["title"])
    results = []

    for title in subjects:
        raw_title = title.get_text().strip()
        body = []
        candidate = title
        while True:
            candidate = candidate.next_sibling
            if candidate is None or candidate.name in rule["stop"]:
                break
            body.append(candidate)

        entry = {
            "title": raw_title,
            "plaintext": [
                elem.get_text() if isinstance(elem, bs4.Tag) else str(elem)
                for elem in body
            ],
            "html": "".join(
                elem.prettify() if isinstance(elem, bs4.Tag) else str(elem)
                for elem in body
            ),
            "url": url,
        }
        entry.update(info)
        entry.update(rule["info"])
        results.append(entry)

    return results


def get_soup(url, cfg, browser):

    if cfg["scraper"] == "selenium":
        wait_time = 4  # Kind of arbitrary, but if internet is slow page doesnt render on time because of js

        browser.get(url)
        # Make sure the page loads, might want to fix this for something more robust
        time.sleep(wait_time)
        html = browser.page_source
        if html:
            soup = BeautifulSoup(html, "lxml")
        else:
            log.info("scraping didn't work for ", url=url)
            return None

    elif cfg["scraper"] == "requests":
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
    return soup


def extract_sections(url, info, cfg, browser, translated=False):
    soup = get_soup(url, cfg, browser)

    if not translated and "translate" in cfg:
        new_url = soup.select_one(cfg["translate"])
        if not new_url:
            return [], []
        new_url = urljoin(url, new_url["href"])
        info["url"] = new_url
        return extract_sections(new_url, info, cfg, browser, True)

    log.info("scraping", urlkey=info["urlkey"], url=url)

    results = []

    for rule in cfg["selectors"]:
        method = globals()[f"rule_{rule['method']}"]
        results.extend(method(soup, info, url, rule))

    return results, soup


def run(yaml_filename, outdir="covidfaq/scrape/", site=None):
    """Scrape websites for information."""

    with open(yaml_filename, "r") as stream:
        sites = load(stream, Loader=yaml.FullLoader)
    now = datetime.now()
    outdir = os.path.join(outdir, now.strftime("%Y%m%d_%I%M"))

    # Initialize broswer for selenium
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--remote-debugging-port=9222")
    browser = webdriver.Chrome(options=options)

    # Scrape sites
    results = []
    soups = []
    for sitename, sitecfg in sites.items():
        if site and site != sitename:
            continue
        for i, url in enumerate(sitecfg["urls"]):
            if isinstance(url, dict):
                urlinfo = dict(url["info"])
                url = url["url"]
            else:
                urlinfo = {}
            urlinfo.setdefault("urlkey", str(i))
            urlinfo["urlkey"] = f"{sitename}-{urlinfo['urlkey']}"
            info = {**sitecfg["info"], **urlinfo, "time": str(now), "url": url}
            result, soup = extract_sections(url, info, sitecfg, browser)
            if result:
                results += result
                soups.append(soup)

    browser.quit()

    # Save scraping results to json files
    files = defaultdict(dict)
    # remap dict key names, for legacy
    for entry in results:
        d = files[entry["urlkey"]]
        d["document_URL"] = entry["url"]
        del entry["urlkey"]
        d[entry["title"]] = entry
    os.makedirs(outdir, exist_ok=True)
    for (filename, data), soup in zip(files.items(), soups):
        filename_json = os.path.join(outdir, filename + ".json")
        page_to_json(data, filename_json)
        filename_html = os.path.join(outdir, filename + ".html")
        soup_to_html(filename_html, soup)

    # Convert scrape results to the bert_reranker format
    # TODO: when there will be more provinces + languages supported, this will need to be updated
    PASSAGES_OUTDIR = os.path.join(outdir, "bert_reranker_format/")
    source = "quebec"
    lang = "en"

    passages = scrapes_to_passages(outdir, source, lang, is_faq=True)
    os.makedirs(PASSAGES_OUTDIR, exist_ok=True)
    dump_passages(
        passages,
        fname=os.path.join(
            PASSAGES_OUTDIR, "source_" + lang + "_faq" + "_passages.json"
        ),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", help="list of sites to scrape", required=True)
    parser.add_argument(
        "--outdir", help="where to save scrapes", default="covidfaq/scrape/"
    )
    args = parser.parse_args()

    run(args.sites, args.outdir)
