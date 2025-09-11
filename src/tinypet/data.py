import xarray as xr

from tinypet.core import Source


class XarraySource(Source):

    def __init__(self, dataset: xr.Dataset, dim: str):
        self.dataset = dataset
        self.dim = dim
        # TODO allow returning lazy or loaded data

    def __getitem__(self, key):
        return self.dataset.sel({self.dim: key})

    @property
    def index(self):
        return self.dataset[self.dim]
