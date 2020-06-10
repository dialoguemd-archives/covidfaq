from covidfaq.scrape import scrape
from covidfaq.k8s import rollout_restart

if __name__ == "__main__":
    scrape.run("covidfaq/scrape/quebec-sites.yaml", "covidfaq/scrape/")
    rollout_restart()
