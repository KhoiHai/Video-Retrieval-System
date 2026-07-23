import os
import cv2 

class Frame:

    def __init__(self, frame_idx):
        # Metadata
        self.frame_idx = frame_idx

        # Frame information
        self.CLIPembedding = None

    def __repr__(self):
        return f"Frame({self.frame_idx})"