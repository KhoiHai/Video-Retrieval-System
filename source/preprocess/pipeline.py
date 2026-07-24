# Entity
from source.entity.video import Video
from source.entity.scene import Scene

# Keyframe Extraction
from source.preprocess.keyframe_extraction.scene_detector import SceneDetector
from source.preprocess.keyframe_extraction.candidate_extractor import CandidateExtractor
from source.preprocess.keyframe_extraction.keyframe_extractor import KeyframeExtractor

# Storage
from source.storage.media.local_media_storage import LocalMediaStorage

# Model 
from source.models.TransNetV2.inference import TransNetV2
from source.models.CLIP.clip import CLIP

class PreprocessingPipeline:

    def __init__(self):
        self.scene_detector = SceneDetector(model = TransNetV2(), threshold = 0.75)
        self.candidate_extractor = CandidateExtractor(blur_threshold = 110, duplicate_threshold = 0.7)
        self.keyframe_extractor = KeyframeExtractor(clip_model = CLIP(), similarity_threshold = 0.83, max_frames_per_segment = 6)
        self.media_storage = LocalMediaStorage("data/keyframe")

    def run(self, video_path):
        video = Video(video_path = video_path)
        self.scene_detector.detect(video)
        self.candidate_extractor.extract(video)
        self.keyframe_extractor.extract(video)
        self.media_storage.save(video)

        return video