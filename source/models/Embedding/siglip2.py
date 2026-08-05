from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

class SigLIP2:

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: str | None = None,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(
        self,
        images: List[Image.Image],
        normalize: bool = True,
    ) -> torch.Tensor:

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        embeddings = self.model.get_image_features(**inputs).pooler_output

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings


    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:

        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        embeddings = self.model.get_text_features(**inputs).pooler_output

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings