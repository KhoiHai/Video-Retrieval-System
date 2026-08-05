from abc import ABC, abstractmethod

from source.entity.video import Video


class VectorStorage(ABC):

    @abstractmethod
    def save(self, video: Video):
        pass

    @abstractmethod
    def load(self, video: Video):
        pass