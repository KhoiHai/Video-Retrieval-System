import os

import numpy as np

from source.entity.video import Video
from source.storage.vector.vector_storage import VectorStorage


class LocalVectorStorage(VectorStorage):

    def __init__(self, root_dir: str = "data/vector"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def save(self, video: Video):

        embeddings = []

        for scene in video.scenes:
            for frame in scene.frames:

                if frame.semantic_embedding is None:
                    continue

                embeddings.append(
                    frame.semantic_embedding.cpu().numpy()
                )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        save_path = os.path.join(
            self.root_dir,
            f"{video.video_id}.npy"
        )

        np.save(save_path, embeddings)

    def load(self, video: Video):

        load_path = os.path.join(
            self.root_dir,
            f"{video.video_id}.npy"
        )

        return np.load(load_path)