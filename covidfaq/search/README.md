### Using Elastic Search locally

Install elastic search on your system using the instructions [here]('https://www.elastic.co/guide/en/elastic-stack-get-started/current/get-started-elastic-stack.html'). Start a local server, and test it works by querying it with
`curl http://127.0.0.1:9200`.

Next, set your environment variables so ES runs locally (it will ping the dev server by default). These need to be reset every time you open a new shell, alternatively you can add them to your bash profile.

`export elastic_search_host='localhost'`
`export elastic_search_port='9200'`

Next, scrape websites and build the index using:

`poetry run python scripts/scraper.py`

You should now be able to ping ES locally. 
You can use `covidfaq/search/search_index.py` as a reference to ping ES for some sample answers to questions.

If you want to use ES running on the dev server, you can use:

`export elastic_search_host='es-covidfaq.dev.dialoguecorp.com'`
`export elastic_search_port='443'`
