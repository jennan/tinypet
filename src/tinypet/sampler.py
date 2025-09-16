from typing import Any
from collections.abc import Sequence
from functools import cached_property

import numpy as np
import pandas as pd
import xarray as xr

from tinypet.core import Source, Step, make_step_builder, is_time_index


class _DateRange(Step):
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


DateRange = make_step_builder(_DateRange)


class _Shuffle(Step):
    def __init__(self, source: Source, seed=None):
        super().__init__(source)
        self._rng = np.random.default_rng(seed)

    @property
    def index(self):
        index = self.source.index
        random_index = self._rng.permutation(index)
        return random_index


Shuffle = make_step_builder(_Shuffle)


class _Batch(Step):
    def __init__(self, source: Source, offsets: Sequence[Any]):
        if is_time_index(source.index):
            offsets = pd.to_timedelta(offsets)
        super().__init__(source)
        self.offsets = offsets

    @cached_property
    def index(self):
        assert isinstance(self.source.index, xr.DataArray)
        dim = f"{self.source.index.dims[0]}_2"
        mask = (
            (self.source.index + xr.DataArray(self.offsets, dims=dim))
            .isin(self.source.index)
            .all(dim=dim)
        )
        return self.source.index[mask]

    def get(self, key):
        samples = [self.source.get(key + offset) for offset in self.offsets]
        return tuple(samples)


Batch = make_step_builder(_Batch)


class _XBatch(_Batch):
    def __init__(self, source: Source, offsets: Sequence[Any], dim: str):
        super().__init__(source, offsets)
        self.dim = dim

    def get(self, key):
        samples = super().get(key)
        return xr.concat(samples, dim=self.dim, join="exact")


XBatch = make_step_builder(_XBatch)


class _NBatch(_Batch):
    def __init__(self, source: Source, offsets: Sequence[Any], axis: int = 0):
        super().__init__(source, offsets)
        self.axis = axis

    def get(self, key):
        samples = super().get(key)
        return np.stack(samples, axis=self.axis)


NBatch = make_step_builder(_NBatch)
