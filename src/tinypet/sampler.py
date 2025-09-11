import numpy as np
import pandas as pd

from tinypet.core import Source, Step, make_step_builder


@make_step_builder
class DateRange(Step):

    def __init__(self, source: Source, start=None, end=None):
        super().__init__(source)
        self.start = start
        self.end = end

    @property
    def index(self):
        full_index = pd.to_datetime(self.source.index)
        mask = True
        if self.start is not None:
            mask &= full_index >= self.start
        if self.end is not None:
            mask &= full_index <= self.end
        index = full_index[mask]
        return index


@make_step_builder
class Shuffle(Step):

    def __init__(self, source: Source, seed = None):
        super().__init__(source)
        self._rng = np.random.default_rng(seed)

    @property
    def index(self):
        index = self.source.index
        random_index = self._rng.permutation(index)
        return random_index
