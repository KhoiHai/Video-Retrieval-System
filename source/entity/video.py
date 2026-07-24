import os 
import json
import cv2
from source.entity.scene import Scene

class Video:

    def __init__(self, video_path):
        # Video Identification
        self.video_path = video_path
        self.video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Video Scene
        self.scenes: list[Scene] = []

        # Video Metadata
        self.__load_metadata()

    def __load_metadata(self):
        # Checking the existence of video
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"[PREPROCESSING] Video is not found in {self.video_path}"
            )
        print(f"[PREPROCESSING] Found video: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)

        # Checking whether the video can be opened or not
        if not cap.isOpened():
            raise RuntimeError(
                f"[PREPROCESSING] Cannot open the video: {self.video_path}"
            )
        print(f"[PREPROCESSING] Video is opened successfully")

        # Video metadata
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps
        
        cap.release()

    def num_scene(self):
        return len(self.scenes)

    def num_keyframe(self):
        count = 0
        for scene in self.scenes:
            count += scene.num_frames
        return count

    def clear_scenes(self):
        self.scenes.clear()

    def get_scene(self, scene_id):
        for scene in self.scenes:
            if scene_id == scene.scene_id:
                return scene
        return None

    def add_scene(self, scene: Scene):
        self.scenes.append(scene)
    
    def to_dict(self):
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "keyframe_count": self.num_keyframe(),
            "width": self.width,
            "height": self.height,
            "duration": self.duration
        }