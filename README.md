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

- replace `__getitem__` with a `.get` method
  - use `__getitem__` for key chech (slower/safer)
  - use `get` for direct query (no checks, faster)
  - use `get` inside `Step` methods
- require a `__contains__` method to check if key is valid
- use `partial(NewBuilder, <class>)` instead of `make_step_build`
