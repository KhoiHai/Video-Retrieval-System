import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from source.models.Embedding.siglip2 import SigLIP2


class SemanticRetriever:

    def __init__(
        self,
        embedding_path: str,
        metadata_path: str,
        embedding_model
    ):
        self.embedding_model = embedding_model

        print("[RETRIEVAL] Loading embeddings...")
        self.embeddings = torch.from_numpy(
            np.load(embedding_path)
        ).float()

        self.embeddings = torch.nn.functional.normalize(
            self.embeddings,
            dim=1
        )

        print(self.embeddings.shape)

        print("[RETRIEVAL] Loading metadata...")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.frames = []

        for scene in self.metadata["scenes"]:
            self.frames.extend(scene["frames"])

        print(f"[RETRIEVAL] Loaded {len(self.frames)} keyframes.")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ):

        text_embedding = self.embedding_model.encode_texts([query]).cpu()

        similarity = torch.matmul(
            self.embeddings,
            text_embedding.squeeze(0)
        )

        values, indices = torch.topk(similarity, top_k)

        results = []

        for score, idx in zip(values.tolist(), indices.tolist()):

            frame = self.frames[idx]

            results.append(
                {
                    "score": score,
                    "frame_idx": frame["frame_idx"],
                    "image_uri": frame["image_uri"],
                }
            )

        return results

    def visualize(self, results):

        plt.figure(figsize=(18,4))

        for i, result in enumerate(results):

            image = cv2.imread(result["image_uri"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(results), i + 1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(
                f'{result["frame_idx"]}\n'
                f'{result["score"]:.3f}'
            )

        plt.show()