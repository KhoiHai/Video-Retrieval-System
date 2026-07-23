import numpy as np
import os
import cv2

from source.entity.video import Video
from source.entity.scene import Scene

class SceneDetector():

    def __init__(self, model, threshold = 0.75):
        self.model = model
        self.threshold = threshold

    def detect(self, video: Video):
        print(f"[PREPROCESSING] Detecting video {video.video_id} scenes")

        # Transition Segment
        transition_segment = self.__extract_transition_segment(video.video_path)

        # Build video scenes
        self.__build_scene(video, transition_segment)

        print(f"[PREPROCESSING] Detected {video.num_scene()} scenes")

    def __extract_transition_segment(self, video_path):
        print(f"[PREPROCESSING] Extracting transition frame using threshold {self.threshold} from video with given path {video_path}")

        # Model inference
        _, _, all_frame_predictions = self.model.predict_video(video_path)
        transition_frames = np.where(all_frame_predictions > self.threshold)[0]

        # Initialization
        transition_segment = []
        start_flag = transition_frames[0]
        end_flag = transition_frames[0]

        for frame in transition_frames[1:]:
            if (end_flag + 1 == frame):
                end_flag = frame
            else:
                transition_segment.append((int(start_flag), int(end_flag)))
                start_flag = frame
                end_flag = frame

        transition_segment.append((int(start_flag), int(end_flag)))

        return transition_segment

    def __build_scene(self, video: Video, transition_segment):
        # Initialization
        video.clear_scenes()
        scene_id = 0
        flag = 0

        for start, end in transition_segment:
            if (flag <= (start - 1)):
                scene = Scene(scene_id = scene_id, start_frame_idx = flag, end_frame_idx = start - 1)
                video.add_scene(scene)
                scene_id += 1
            flag = end + 1

        if flag < video.frame_count:
            scene = Scene(scene_id = scene_id, start_frame_idx = flag, end_frame_idx = video.frame_count - 1)
            video.add_scene(scene)
