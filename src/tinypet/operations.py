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

    def get(self, key):
        data = self.source.get(key)
        data = self.apply(data)
        return data


class _Identity(SimpleOp):
    def apply(self, data):
        return data

    @property
    def undo_builder(self):
        return Identity()


Identity = make_step_builder(_Identity)


class _Rename(SimpleOp):
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


Rename = make_step_builder(_Rename)


class _Mean(SimpleOp):
    def apply(self, data: xr.Dataset) -> xr.Dataset:
        return data.mean(*self.args, **self.kwargs)


Mean = make_step_builder(_Mean)


class _Compute(SimpleOp):
    def apply(self, data: xr.Dataset) -> xr.Dataset:
        return data.compute(*self.args, **self.kwargs)

    @property
    def undo_builder(self):
        return Identity()


Compute = make_step_builder(_Compute)


class _ToUnstackedDataset(SimpleOp):
    def apply(self, data: xr.Dataset) -> xr.DataArray:
        return data.to_unstacked_dataset(*self.args, **self.kwargs)


ToUnstackedDataset = make_step_builder(_ToUnstackedDataset)


class _ToStackedArray(SimpleOp):
    def apply(self, data: xr.Dataset) -> xr.DataArray:
        return data.to_stacked_array(*self.args, **self.kwargs)

    @property
    def undo_builder(self):
        if len(self.args) == 0:
            dim = self.kwargs["dim"]
        else:
            dim = self.args[0]
        return ToUnstackedDataset(dim)


ToStackedArray = make_step_builder(_ToStackedArray)


class _Sel(SimpleOp):
    def apply(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        return data.sel(*self.args, **self.kwargs)


Sel = make_step_builder(_Sel)


class _Merge(SimpleOp):
    def apply(self, objects) -> xr.Dataset:
        return xr.merge(objects, *self.args, **self.kwargs)


Merge = make_step_builder(_Merge)


class _Concat(SimpleOp):
    def apply(self, objects) -> xr.Dataset:
        return xr.concat(objects, *self.args, **self.kwargs)


Concat = make_step_builder(_Concat)


class _Select(Step):
    def __init__(self, source, varnames):
        super().__init__(source)
        self.varnames = varnames

    def get(self, key):
        data = self.source.get(key)[self.varnames]
        return data


Select = make_step_builder(_Select)


class _ToDataArray(Step):
    def __init__(self, source, coords):
        super().__init__(source)
        self.coords = coords

    def get(self, key):
        arr = self.source.get(key)
        data = xr.DataArray(arr, coords=self.coords)
        if (index_coord := self.source.index.name) in data.coords:
            data.coords[index_coord] = data.coords[index_coord] + key
        return data


ToDataArray = make_step_builder(_ToDataArray)


class _ToNumpy(Step):
    def __init__(self, source):
        super().__init__(source)

    def get(self, key):
        darr = self.source.get(key)
        return darr.data

    @property
    def undo_builder(self):
        assert isinstance(self.index, xr.DataArray)
        index0 = self.source.index[0]
        sample0 = self.source.get(index0)
        coords = sample0.coords
        if (index_coord := self.source.index.name) in coords:
            coords[index_coord] = coords[index_coord] - index0
        return ToDataArray(coords)


ToNumpy = make_step_builder(_ToNumpy)


class _Op(SimpleOp):
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
        return Op(self.undo_func, *self.args, undo_func=self.apply_func, **self.kwargs)


class Op(StepBuilder):
    def __init__(self, apply_func, *args, undo_func=None, **kwargs):
        super().__init__(_Op, apply_func, *args, undo_func=undo_func, **kwargs)
