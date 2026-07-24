from source.storage.media.media_storage import MediaStorage
from source.entity.video import Video
from source.entity.scene import Scene
from source.entity.frame import Frame

import os
import cv2

class LocalMediaStorage(MediaStorage):

    def __init__(self, root_dir = "data/keyframe"):
        self.root_dir = root_dir

    def save(self, video: Video):
        print(f"[STORAGE] Saving keyframes of {video.video_id}")

        # Create the output directory base on the video
        video_dir = os.path.join(self.root_dir, video.video_id)
        os.makedirs(video_dir, exist_ok = True)

        # Read the videp
        cap = cv2.VideoCapture(video.video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open {video.video_path}"
            )
        
        # Traverse through every scene and keyframe inside the video
        for scene in video.scenes:
            for frame in scene.frames:
                # Read the image
                image = self.__read_frame(cap, frame.frame_idx)
                if image is None:
                    continue

                # Save the image from given path
                image_path = os.path.join(video_dir, f"frame_{frame.frame_idx:06d}.jpg")
                cv2.imwrite(image_path, image)

                # Update the frame.image_uri
                frame.set_uri(image_path)

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