import json
import yaml
import argparse
from yaml import load
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from urllib.parse import urljoin

import bs4
import requests
import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)


def remove_html_tags(data):
    p = re.compile(r"<.*?>")
    return p.sub("", data)


def soup_to_html(filename, soup):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(soup))


def page_to_json(page_contents, fname):
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

        if exclude["title"] and re.search(exclude["title"], raw_title):
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


def extract_sections(url, info, cfg, translated=False):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    if not translated and "translate" in cfg:
        new_url = soup.select_one(cfg["translate"])
        if not new_url:
            return [], []
        new_url = urljoin(url, new_url["href"])
        return extract_sections(new_url, info, cfg, True)

    log.info("scraping", urlkey=info["urlkey"], url=url)

    results = []

    for rule in cfg["selectors"]:
        method = globals()[f"rule_{rule['method']}"]
        results.extend(method(soup, info, url, rule))

    return results, soup


def run(yaml_filename, outdir="covidfaq/scrape", formatting="old", site=None):
    """Scrape websites for information."""

    with open(yaml_filename, "r") as stream:
        sites = load(stream, Loader=yaml.FullLoader)
    now = str(datetime.now())

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
            info = {**sitecfg["info"], **urlinfo, "time": now, "url": url}
            result, soup = extract_sections(url, info, sitecfg)
            if result:
                results += result
                soups.append(soup)

    if formatting == "old":
        files = defaultdict(dict)
        # change key names
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

    elif formatting == "new":
        outfile = out or "scrape_results.json"
        page_to_json(results, outfile)

    else:
        print(f"Unknown format: {format}")
        sys.exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", help="list of sites to scrape", required=True)
    parser.add_argument(
        "--outdir", help="where to save scrapes", default="covidfaq/scrape"
    )
    args = parser.parse_args()

    run(args.sites, args.outdir)
