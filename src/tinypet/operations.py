from abc import abstractmethod

import xarray as xr

from tinypet.core import Step, StepBuilder, make_step_builder


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


@make_step_builder
class ToUnstackedDataset(SimpleOp):
    def apply(self, data: xr.Dataset) -> xr.DataArray:
        return data.to_unstacked_dataset(*self.args, **self.kwargs)


@make_step_builder
class ToStackedArray(SimpleOp):
    def apply(self, data: xr.Dataset) -> xr.DataArray:
        return data.to_stacked_array(*self.args, **self.kwargs)

    @property
    def undo_builder(self):
        if len(self.args) == 0:
            dim = self.kwargs["dim"]
        else:
            dim = self.args[0]
        return ToUnstackedDataset(dim)


@make_step_builder
class Sel(SimpleOp):
    def apply(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        return data.sel(*self.args, **self.kwargs)


@make_step_builder
class Select(Step):
    def __init__(self, source, varnames):
        super().__init__(source)
        self.varnames = list(varnames)

    def __getitem__(self, key):
        data = self.source[key][self.varnames]
        return data


@make_step_builder
class ToDataArray(Step):
    def __init__(self, source, coords):
        super().__init__(source)
        self.coords = coords

    def __getitem__(self, key):
        arr = self.source[key]
        data = xr.DataArray(arr, coords=self.coords)
        return data


@make_step_builder
class ToNumpy(Step):
    def __getitem__(self, key):
        darr = self.source[key]
        return darr.data

    @property
    def undo_builder(self):
        coords = next(iter(self.source)).coords
        return ToDataArray(coords)


class Op(StepBuilder):
    def __init__(self, apply_func, *args, undo_func=None, **kwargs):
        super().__init__(FunctionOp, apply_func, *args, undo_func=undo_func, **kwargs)


class FunctionOp(SimpleOp):
    def __init__(self, source, apply_func, *args, undo_func=None, **kwargs):
        super().__init__(source, *args, **kwargs)
        self.apply_func = apply_func
        self.undo_func = undo_func

    def apply(self, sample):
        return self.apply_func(sample, *self.args, **self.kwargs)

    @property
    def undo_builder(self):
        if self.undo_func is None:
            raise NotImplementedError("Missing undo function.")
        # TODO document that undo_func gets the same parameters as apply_func
        return Op(self.undo_func, *self.args, undo=self.apply_func, **self.kwargs)
