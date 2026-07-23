import os
import cv2
from source.entity.frame import Frame

class Scene:

    def __init__(self, scene_id, start_frame_idx, end_frame_idx):
        # Scene Identification
        self.scene_id = scene_id

        # Scene Metadata
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx

        # Scene frames
        self.frames: list[Frame] = []

    def add_frame(self, frame: Frame):
        self.frames.append(frame)

    def clear_frames(self):
        self.frames.clear()

    def num_frames(self):
        return len(self.frames)

    def __str__(self):
        return(
            f"Scene {self.scene_id}: ({self.start_frame_idx}, {self.end_frame_idx})\n"
            f"Candidate frames: {self.frames}"
        )

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "start_frame_idx": self.start_frame_idx,
            "end_frame_idx": self.end_frame_idx
        }
