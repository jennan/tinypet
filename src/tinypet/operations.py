from abc import abstractmethod

import xarray as xr

from tinypet.core import Step, make_step_builder


class SimpleOp(Step):

    def __init__(self, source, *args, **kwargs):
        super().__init__(source)
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def apply(self, sample):
        pass

    def __getitem__(self, key):
        data = self.source[key]
        data = self.apply(data)
        return data


@make_step_builder
class Rename(SimpleOp):

    def apply(self, data: xr.Dataset) -> xr.Dataset:
        return data.rename(*self.args, **self.kwargs)


@make_step_builder
class Mean(SimpleOp):

    def apply(self, data: xr.Dataset) -> xr.Dataset:
        return data.mean(*self.args, **self.kwargs)
