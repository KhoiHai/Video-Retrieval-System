from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

class EmbeddingModel(ABC):

    @abstractmethod
    def encode_images(
        self,
        images: list[Image.Image]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def encode_texts(
        self,
        texts: list[str]
    ) -> np.ndarray:
        pass