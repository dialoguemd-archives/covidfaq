# covidfaq

## Setup

We use `poetry`. Run `poetry install` at the root of the repo.

### Git LFS

The model (*.ckpt) is tracked with [Git LFS](https://git-lfs.github.com/) (Large File System)

## Evaluator

The code for the evaluator can be found under `covidfaq/evaluating`.

### Run an evaluation

The main script is `evaluator.py`. It needs to be pointed to a json file
containing the evaluation data and to know which model should be evaluated.
Optionally, it accept a config file to initialize the model. E.g.,

    poetry run python covidfaq/evaluating/evaluator.py
      --test-data=covidfaq/evaluating/faq_eval_data.json 
      --model-type=embedding_based_reranker
      --config=[...]/config.yaml

    poetry run python covidfaq/evaluating/evaluator.py
      --test-data=covidfaq/evaluating/faq_eval_data.json 
      --model-type=cheating_model

### How to evaluate a new model

To use the evaluator with a new model, two modifications must be done:
 - create a new class (under `covidfaq/evaluating/model`) that implements
 the interface `covidfaq/evaluating/model/model_evaluation_interface.py`.
 See the doc in this interface for more info on the two methods to implement.
 Note that any information you need to initialize your model must be passed
 in the config file that is passed with `--config=..../config.yaml` to the
 evaluator. For example, the saved model weight location can be specified here.
 - Add an if in the `evaluator.py` to accept your model (and to load the
 proper class that you implemented in the point above).