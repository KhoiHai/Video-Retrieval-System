from abc import ABC, abstractmethod
from source.entity.video import Video

class MediaStorage(ABC):

    @abstractmethod
    def save(self, video: Video):
        pass