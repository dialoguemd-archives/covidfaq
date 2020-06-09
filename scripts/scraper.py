from covidfaq.scrape import scrape

if __name__ == "__main__":
    scrape.run("covidfaq/scrape/quebec-sites.yaml", "covidfaq/scrape/")
