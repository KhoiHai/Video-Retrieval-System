import numpy as np
import cv2

from source.entity.video import Video
from source.entity.scene import Scene
from source.entity.frame import Frame

class CandidateExtractor:

    def __init__(self, sampling_step = 15, blur_threshold = 120, duplicate_threshold = 0.7):
        # Hyperparameter for extracting candidates
        self.sampling_step = sampling_step
        self.blur_threshold = blur_threshold
        self.duplicate_threshold = duplicate_threshold

    def extract(self, video: Video):
        print(f"[PREPROCESSING] Extract candidate frames from {video.video_id}")

        cap = cv2.VideoCapture(video.video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"[PREPROCESSING] Cannot open {video.video_path}"
            )

        self.sampling_step = int(video.fps / 2)

        for scene in video.scenes:
            scene.clear_frames()

            self.__extract_scene_candidates(
                cap,
                scene
            )

            print(
                f"[PREPROCESSING] Scene {scene.scene_id}: "
                f"{scene.num_frames()} candidate frames"
            )

        cap.release()


    def __blur_score(self, frame):
        # Convert to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def __histogram_similarity(self, frame1, frame2):
        # Convert to HSV
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # 3D HSV Histogram (512 bins)
        hist1 = cv2.calcHist(
            [hsv1],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )

        hist2 = cv2.calcHist(
            [hsv2],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )

        # Normalization
        cv2.normalize(hist1, hist1, norm_type=cv2.NORM_L1)
        cv2.normalize(hist2, hist2, norm_type=cv2.NORM_L1)

        return cv2.compareHist(
            hist1,
            hist2,
            cv2.HISTCMP_CORREL
        )

    def __extract_scene_candidates(self, cap, scene: Scene):
        # Loop through every single frame inside the scene
        selected_images = []

        for frame_idx in range(scene.start_frame_idx, scene.end_frame_idx + 1, self.sampling_step):
            # Read the image 
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, image = cap.read()
            if not ret:
                continue

            # Blurry elimination
            if self.__blur_score(image) < self.blur_threshold:
                continue

            # Keep the first valid frame or image
            if len(selected_images) == 0:
                selected_images.append(image)
                scene.add_frame(
                    Frame(frame_idx)
                )
                continue

            duplicated = False

            for previous in selected_images:
                similarity = self.__histogram_similarity(
                    previous,
                    image
                )

                if similarity >= self.duplicate_threshold:
                    duplicated = True
                    break

            if duplicated:
                continue

            selected_images.append(image)

            scene.add_frame(
                Frame(frame_idx)
            )