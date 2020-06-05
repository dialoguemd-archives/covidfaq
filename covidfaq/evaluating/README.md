# How to use `evaluating`

Make sure to download the correct evaluation or test JSON files from the data repository, and place it in a path. Make sure you `ls`'d  into `covidfaq/covidfaq/evaluating/`.

Then, run the following command:

```
poetry run python evaluator.py --test-data=/path/to/validation_file.json --model-type=your_model
```

where `your_model` is one of the models in `covidfaq.evaluating.model`. For more info, run:
```
poetry run python evaluator.py --help
```
