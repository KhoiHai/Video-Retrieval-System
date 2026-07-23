import os
import cv2

class Scene:

    def __init__(self, scene_id, start_frame_idx, end_frame_idx):
        # Scene Identification
        self.scene_id = scene_id

        # Scene Metadata
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx

    def num_frames(self):
        return self.end_frame_idx - self.start_frame_idx + 1

    def __str__(self):
        return(
            f"Scene {self.scene_id}: ({self.start_frame_idx}, {self.end_frame_idx})\n"
        )

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "start_frame_idx": self.start_frame_idx,
            "end_frame_idx": self.end_frame_idx
        }
