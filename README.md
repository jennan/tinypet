Missing features

- branching
- caching
- graph representation (text and image)
- serialisation/deserialisation (add Step.to_dict/Step.from_dict, leverage pydantic?)
- using it with ML frameworks (sklearn, xgboost, torch, lightning, jax, etc.)
  - how to train a model, i.e. feed data using pipelines
  - use trained model inside a pipeline, for inference
- parallel fetching with joblib
- parallel fetching with multiprocessing (and show pickling issue with lambdas)
