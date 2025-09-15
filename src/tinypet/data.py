from collections.abc import Sequence

import xarray as xr

from tinypet.core import Source


class Seq(Source):
    def __init__(self, data: Sequence):
        # TODO allow attaching an arbitrary index
        self.data = data

    def get(self, key):
        return self.data[key]

    @property
    def index(self):
        return range(len(self.data))


class XarraySource(Source):
    def __init__(self, dataset: xr.Dataset, dim: str):
        self.dataset = dataset
        self.dim = dim

    def get(self, key):
        return self.dataset.sel({self.dim: key})

    @property
    def index(self):
        return self.dataset[self.dim]
