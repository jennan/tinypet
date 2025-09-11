from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Sequence


class Source(ABC):

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        pass

    @property
    @abstractmethod
    def index(self) -> Sequence[Any]:
        pass

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Any:
        for i in self.index:
            yield self[i]


class Step(Source):

    def __init__(self, source):
        self.source = source

    @property
    def index(self):
        return self.source.index

    def __getitem__(self, key):
        # TODO enforce checking if key is in index?
        return self.source[key]


class StepBuilder:

    def __init__(self, step_class, *args, **kwargs):
        self.step_class = step_class
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source: Source) -> Step:
        # TODO handle case of other is a StepBuilder
        return self.step_class(source, *self.args, **self.kwargs)

    def __ror__(self, other: Source) -> Step:
        return self(other)


def make_step_builder(cls):

    class NewBuilder(StepBuilder):

        def __init__(self, *args, **kwargs):
            super().__init__(cls, *args, **kwargs)
    
    return NewBuilder
