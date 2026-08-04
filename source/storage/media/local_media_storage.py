from source.storage.media.media_storage import MediaStorage
from source.entity.video import Video
from source.entity.scene import Scene
from source.entity.frame import Frame

import os
import cv2
from pathlib import Path

class LocalMediaStorage(MediaStorage):

    def __init__(self, root_dir = "data/keyframe"):
        self.root_dir = Path(root_dir)

    def save(self, video: Video):
        print(f"[STORAGE] Saving keyframes of {video.video_id}")

        # Create output directory
        video_dir = self.root_dir / video.video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video.video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open {video.video_path}"
            )

        for scene in video.scenes:
            for frame in scene.frames:

                image = self.__read_frame(cap, frame.frame_idx)
                if image is None:
                    continue

                image_path = video_dir / f"frame_{frame.frame_idx:06d}.jpg"

                # Write down path
                cv2.imwrite(str(image_path), image)
                frame.set_uri(image_path.as_posix())

        cap.release()
        print(f"[STORAGE] Finished saving {video.video_id}")

    def __read_frame(self, cap, frame_idx):
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        success, image = cap.read()
        if not success:
            return None

        return image