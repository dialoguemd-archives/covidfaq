from covidfaq.scrape import scrape
from covidfaq.search import build_index

if __name__ == "__main__":
    scrape.run("covidfaq/scrape/quebec-sites.yaml")
    build_index.run()
