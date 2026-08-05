from abc import ABC, abstractmethod
from typing import List
from source.entity.frame import Frame

class FeatureExtractor(ABC):

    @abstractmethod
    def extract(self, frames: List[Frame]):
        pass