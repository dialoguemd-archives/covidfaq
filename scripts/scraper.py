import coleo

from covidfaq.scrape import scrape
from covidfaq.search import build_index

if __name__ == "__main__":
    with coleo.setvars(sites=coleo.ConfigFile("covidfaq/scrape/quebec-sites.yaml")):
        scrape.run()
    build_index.run()
