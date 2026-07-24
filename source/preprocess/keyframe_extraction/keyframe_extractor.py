import cv2
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from source.entity.video import Video
from source.entity.scene import Scene

class KeyframeExtractor:

    def __init__(self, clip_model, similarity_threshold = 0.83, max_frames_per_segment = 6):
        # Hyperparameter and model for keyframe extraction
        self.clip_model = clip_model
        self.similarity_threshold = similarity_threshold
        self.max_frames_per_segment = max_frames_per_segment

    def extract(self, video: Video):
        print(f"[PREPROCESSING] Extracting keyframes from {video.video_id}")

        # Video reading
        cap = cv2.VideoCapture(video.video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open {video.video_path}"
            )

        for scene in video.scenes:
            if scene.num_frames() == 0:
                continue

            self.__encode_scene(
                cap,
                scene
            )

            chunks = self.__semantic_chunking(scene)
            keyframes = self.__representative_selection(chunks)

            scene.clear_frames()
            for frame in keyframes:
                scene.add_frame(frame)

            print(f"[PREPROCESSING] Scene {scene.scene_id}: {scene.num_frames()} keyframes")

        cap.release()

    def __encode_scene(self, cap, scene: Scene):
        # Loop through every frames
        for frame in scene.frames:
            # Image reading
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame.frame_idx
            )
            ret, image = cap.read()
            if not ret:
                continue

            frame.selection_embedding = (
                self.clip_model.encode(image)
            )

    def __semantic_chunking(self, scene: Scene):
        # Initialization
        chunks = []
        current_chunk = [scene.frames[0]]

        for i in range(1, scene.num_frames()):
            # Initialization
            previous = scene.frames[i - 1]
            current = scene.frames[i]

            similarity = cosine_similarity(
                previous.selection_embedding.reshape(1, -1),
                current.selection_embedding.reshape(1, -1)
            )[0][0]

            if similarity >= self.similarity_threshold:
                current_chunk.append(current)
            else:
                chunks.append(current_chunk)

                current_chunk = [current]

        chunks.append(current_chunk)

        return chunks

    def __representative_selection(self, chunks):
        # Initialization
        keyframes = []

        for chunk in chunks:
            n = len(chunk)
            if n == 0:
                continue
            num_segments = int(np.ceil(n / self.max_frames_per_segment))

            segments = np.array_split(chunk, num_segments)

            for segment in segments:
                if len(segment) == 0:
                    continue

                embeddings = np.stack(
                    [
                        frame.selection_embedding
                        for frame in segment
                    ]
                )

                centroid = embeddings.mean(axis=0)
                distances = np.linalg.norm(embeddings - centroid, axis=1)
                best = np.argmin(distances)
                keyframes.append(segment[best])

        return keyframes