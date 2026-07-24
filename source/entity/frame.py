import os
import cv2 

class Frame:

    def __init__(self, frame_idx):
        # Metadata
        self.frame_idx = frame_idx
        self.image_uri = None

        # Selection embedding
        self.selection_embedding = None  

        # Retrieval features      

    def __repr__(self):
        return f"Frame({self.frame_idx})"

    def set_uri(self, image_uri: str):
        self.image_uri = image_uri
