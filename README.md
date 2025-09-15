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

Other ideas

- use `__contains__` method to check if key is valid, allow overriding for optimisation
- use `partial(NewBuilder, <class>)` instead of `make_step_build`
