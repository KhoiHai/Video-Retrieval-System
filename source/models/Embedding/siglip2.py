from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling


class SigLIP2:

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: str | None = None,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _to_tensor(output) -> torch.Tensor:
        """Xử lý cả 2 trường hợp: tensor thẳng hoặc BaseModelOutputWithPooling."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, BaseModelOutputWithPooling):
            return output.pooler_output
        raise TypeError(f"Không nhận diện được kiểu trả về: {type(output)}")

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], normalize: bool = True) -> torch.Tensor:
        images = [img.convert("RGB") for img in images]

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        embeddings = self._to_tensor(self.model.get_image_features(**inputs))
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    @torch.no_grad()
    def encode_texts(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        texts = [t.lower().strip() for t in texts]

        inputs = self.processor(
            text=texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        embeddings = self._to_tensor(self.model.get_text_features(**inputs))
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings