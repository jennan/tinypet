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
class Identity(SimpleOp):

    def apply(self, data):
        return data

    @property
    def undo_builder(self):
        return Identity()


@make_step_builder
class Rename(SimpleOp):

    def apply(self, data: xr.Dataset) -> xr.Dataset:
        return data.rename(*self.args, **self.kwargs)

    @property
    def undo_builder(self):
        assert len(self.args) <= 1
        if self.args:
            mapping = dict(self.args[0], **self.kwargs)
        else:
            mapping = self.kwargs
        # TODO check if mapping is reversible
        reverse_mapping = {value: key for key, value in mapping.items()}
        return Rename(**reverse_mapping)


@make_step_builder
class Mean(SimpleOp):

    def apply(self, data: xr.Dataset) -> xr.Dataset:
        return data.mean(*self.args, **self.kwargs)


@make_step_builder
class Compute(SimpleOp):

    def apply(self, data: xr.Dataset) -> xr.Dataset:
        return data.compute(*self.args, **self.kwargs)

    @property
    def undo_builder(self):
        return Identity()
