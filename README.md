# covidfaq

## Setup
We use `poetry`. Run `poetry install` at the root of the repo.

## Training a model
For `French`, run: 

```bash
poetry run rasa train --data data/fr/ -d domain.yml --out models/fr -c config.fr.yml
```

For `English`, run: 

```bash
poetry run rasa train --data data/en/ -d domain.yml --out models/en -c config.en.yml
```

## Testing a model
```bash
poetry run rasa test nlu -m <PATH TO YOUR MODEL> --nlu data/<LANG>/
```


## Talking to your bot
```bash
poetry run rasa shell -m <PATH TO YOUR MODEL>
```
