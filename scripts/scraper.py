from covidfaq.scrape import scrape
from covidfaq.search import build_index

if __name__ == "__main__":
    scrape.run()
    build_index.run()
