import cv2
import torch
import open_clip
import numpy as np

from PIL import Image

class CLIP:

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=None,
    ):

        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, image):

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        image = self.preprocess(image)

        image = image.unsqueeze(0).to(self.device)

        embedding = self.model.encode_image(image)

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

        return (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )