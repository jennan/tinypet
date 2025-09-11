Missing features

- branching
- caching
- graph representation (text and image)
- serialisation/deserialisation (add Step.to_dict/Step.from_dict)
- using it with ML frameworks (sklearn, xgboost, torch, lightning, jax, etc.)
  - how to train a model, i.e. feed data using pipelines
  - use trained model inside a pipeline, for inference
- wrap simple function as operation (optionally with an undo function)
- simple repeat of a value if not a Source at beginning of pipeline
- parallel fetching with joblib
- parallel fetching with multiprocessing (and show pickling issue with lambdas)
