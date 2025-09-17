from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union

from numpy.dtypes import DateTime64DType
import pandas as pd
import xarray as xr


def is_time_index(index):
    return isinstance(index, xr.DataArray) and isinstance(index.dtype, DateTime64DType)


class Source(ABC):
    @abstractmethod
    def get(self, key: Any) -> Any:
        pass

    @property
    @abstractmethod
    def index(self) -> Sequence[Any]:
        pass

    def __getitem__(self, key: Any) -> Any:
        if is_time_index(self.index):
            key = pd.to_datetime(key)
        # TODO check speed of "in" for large unsorted index
        if key not in self.index:
            raise KeyError(f"Key '{key}' is not in the index of the source.")
        return self.get(key)

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Any:
        for i in self.index:
            yield self.get(i)

    def __add__(
        self, other: Union["Source", "Builder"]
    ) -> Union["Branch", "BranchBuilder"]:
        # TODO replace with match statement?
        if isinstance(other, BranchBuilder):
            branch = BranchBuilder(self, *other.builders)
        elif isinstance(other, Builder):
            branch = BranchBuilder(self, other)
        elif isinstance(other, Branch):
            branch = Branch(self, *other.sources)
        elif isinstance(other, Source):
            branch = Branch(self, other)
        return branch


class Step(Source):
    def __init__(self, source):
        self.source = source

    def get(self, key):
        return self.source.get(key)

    @property
    def index(self):
        return self.source.index

    @property
    def undo_builder(self) -> "StepBuilder":
        raise NotImplementedError(f"Step {self} cannot be undone.")

    def undo(self, n: int = 1) -> Union["StepBuilder", "VirtualBuilder"]:
        # TODO allow list to explicitely skip some steps
        undo_pipe = self.undo_builder
        current_step = self
        for i in range(1, n):
            current_step = current_step.source
            undo_pipe |= current_step.undo_builder
        return undo_pipe


class Branch(Source):
    # TODO add undo support

    def __init__(self, *sources: Source):
        self.sources = sources

    def get(self, key):
        samples = [source.get(key) for source in self.sources]
        return tuple(samples)

    @property
    def index(self):
        # TODO merge indices
        return self.sources[0].index

    def __add__(
        self, other: Union["Source", "Builder"]
    ) -> Union["Branch", "BranchBuilder"]:
        # TODO replace with match statement?
        if isinstance(other, BranchBuilder):
            branch = BranchBuilder(*self.sources, *other.builders)
        elif isinstance(other, Builder):
            branch = BranchBuilder(*self.sources, other)
        elif isinstance(other, Branch):
            branch = Branch(*self.sources, *other.sources)
        elif isinstance(other, Source):
            branch = Branch(*self.sources, other)
        return branch


class Builder(ABC):
    @abstractmethod
    def build(self, source: Source) -> Step:
        pass

    def __call__(
        self, source: Union[Source, "Builder"]
    ) -> Union[Step, "VirtualBuilder"]:
        # TODO replace with match statement?
        if isinstance(source, Source):
            step = self.build(source)
        elif isinstance(source, VirtualBuilder):
            step = VirtualBuilder(*source.builders, self)
        elif isinstance(source, Builder):
            step = VirtualBuilder(source, self)
        return step

    def __ror__(self, other: Source) -> Step:
        return self(other)

    def __add__(self, other: Union["Source", "Builder"]) -> "BranchBuilder":
        # TODO replace with match statement?
        if isinstance(other, BranchBuilder):
            branch = BranchBuilder(self, *other.builders)
        elif isinstance(other, (Source, Builder)):
            branch = BranchBuilder(self, other)
        return branch


class StepBuilder(Builder):
    def __init__(self, step_class, *args, **kwargs):
        self.step_class = step_class
        self.args = args
        self.kwargs = kwargs

    def build(self, source: Source) -> Step:
        return self.step_class(source, *self.args, **self.kwargs)


class VirtualBuilder(Builder):
    def __init__(self, *builders: Builder):
        self.builders = builders

    def build(self, source: Source) -> Step:
        for builder in self.builders:
            source = builder(source)
        return source


class BranchBuilder(Builder):
    def __init__(self, *builders: Source | Builder):
        self.builders = builders

    def build(self, source: Source) -> Step:
        if isinstance(source, Branch):
            if len(source.sources) != len(self.builders):
                raise ValueError(
                    "Mismatching number of elements in successive branches."
                )
            pairs = zip(source.sources, self.builders)
            steps = [builder.build(source) for source, builder in pairs]
        else:
            steps = []
            for builder in self.builders:
                if isinstance(builder, Source):
                    step = builder
                elif isinstance(builder, Builder):
                    step = builder.build(source)
                steps.append(step)
        return Branch(*steps)

    def __add__(self, other: Union["Source", "Builder"]) -> "BranchBuilder":
        # TODO replace with match statement?
        if isinstance(other, BranchBuilder):
            branch = BranchBuilder(*self.builders, *other.builders)
        elif isinstance(other, (Source, Builder)):
            branch = BranchBuilder(*self.builders, other)
        return branch


def make_step_builder(cls):
    class NewBuilder(StepBuilder):
        def __init__(self, *args, **kwargs):
            super().__init__(cls, *args, **kwargs)

    return NewBuilder
