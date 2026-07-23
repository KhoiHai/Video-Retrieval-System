# Entity
from source.entity.video import Video
from source.entity.scene import Scene

# Keyframe Extraction
from source.preprocess.keyframe_extraction.scene_detector import SceneDetector

# Model 
from source.models.TransNetV2.inference import TransNetV2

class PreprocessingPipeline:

    def __init__(self):
        self.scene_detector = SceneDetector(model = TransNetV2(), threshold = 0.75)

    def run(self, video_path):
        video = Video(video_path = video_path)
        self.scene_detector.detect(video)

        return video