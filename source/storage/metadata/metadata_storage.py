from abc import ABC, abstractmethod
from source.entity.video import Video

class MetadataStorage(ABC):

    @abstractmethod
    def save(self, video: Video):
        pass