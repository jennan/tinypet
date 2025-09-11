from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union


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


class StepBuilder:

    def __init__(self, step_class, *args, **kwargs):
        self.step_class = step_class
        self.args = args
        self.kwargs = kwargs

    def __call__(self, source: Union[Source, "StepBuilder", "VirtualBuilder"]) -> Union[Step, "VirtualBuilder"]:
        if isinstance(source, StepBuilder):
            step = VirtualBuilder(source, self)
        elif isinstance(source, VirtualBuilder):
            step = VirtualBuilder(*source.builders, self)
        elif isinstance(source, Source):
            step = self.step_class(source, *self.args, **self.kwargs)
        return step

    def __ror__(self, other: Source) -> Step:
        return self(other)


class VirtualBuilder:

    def __init__(self, *builders):
        self.builders = builders

    def __call__(self, source):
        if isinstance(source, StepBuilder):
            step = VirtualBuilder(source, *self.builders)
        elif isinstance(source, VirtualBuilder):
            step = VirtualBuilder(*source.builders, *self.builders)
        elif isinstance(source, Source):
            for builder in self.builders:
                source = builder(source)
            step = source
        return step

    def __ror__(self, other: Source) -> Step:
        return self(other)


def make_step_builder(cls):

    class NewBuilder(StepBuilder):

        def __init__(self, *args, **kwargs):
            super().__init__(cls, *args, **kwargs)
    
    return NewBuilder
