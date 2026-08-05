from PIL import Image

from source.entity.video import Video
from source.models.Embedding.embedding_models import EmbeddingModel


class SemanticExtractor:

    def __init__(self, embedding_model: EmbeddingModel, batch_size: int = 32):
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def extract(self, video: Video) -> None:

        frames = []

        for scene in video.scenes:
            frames.extend(scene.frames)

        for start in range(0, len(frames), self.batch_size):

            batch_frames = frames[start:start + self.batch_size]

            images = [
                Image.open(frame.image_uri).convert("RGB")
                for frame in batch_frames
            ]

            embeddings = self.embedding_model.encode_images(images)

            embeddings = embeddings.cpu()

            for frame, embedding in zip(batch_frames, embeddings):
                frame.set_semantic_embedding(embedding)