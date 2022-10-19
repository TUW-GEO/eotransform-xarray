from abc import abstractmethod
from typing import Mapping, Any


class Storage:
    @abstractmethod
    def exists(self) -> bool:
        ...

    @abstractmethod
    def load(self) -> Mapping[str, Any]:
        ...

    @abstractmethod
    def save(self, data: Mapping[str, Any]) -> None:
        ...
