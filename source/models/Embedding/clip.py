from typing import List

import torch
import torch.nn.functional as F
import open_clip

from PIL import Image


class CLIPS:

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        (
            self.model,
            _,
            self.preprocess,
        ) = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(
        self,
        images: List[Image.Image],
        normalize: bool = True,
    ) -> torch.Tensor:

        inputs = torch.stack(
            [
                self.preprocess(image)
                for image in images
            ]
        ).to(self.device)

        embeddings = self.model.encode_image(inputs)

        if normalize:
            embeddings = F.normalize(
                embeddings,
                dim=-1,
            )

        print(embeddings.shape)

        return embeddings

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:

        tokens = self.tokenizer(texts).to(self.device)

        embeddings = self.model.encode_text(tokens)

        if normalize:
            embeddings = F.normalize(
                embeddings,
                dim=-1,
            )

        print(embeddings.shape)

        return embeddings