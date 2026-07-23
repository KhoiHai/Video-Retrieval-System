# Entity
from source.entity.video import Video
from source.entity.scene import Scene

# Keyframe Extraction
from source.preprocess.keyframe_extraction.scene_detector import SceneDetector
from source.preprocess.keyframe_extraction.candidate_extractor import CandidateExtractor

# Model 
from source.models.TransNetV2.inference import TransNetV2

class PreprocessingPipeline:

    def __init__(self):
        self.scene_detector = SceneDetector(model = TransNetV2(), threshold = 0.75)
        self.candidate_extractor = CandidateExtractor(blur_threshold = 110, duplicate_threshold = 0.7)

    def run(self, video_path):
        video = Video(video_path = video_path)
        self.scene_detector.detect(video)
        self.candidate_extractor.extract(video)

        return video