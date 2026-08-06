# Entity
from source.entity.video import Video
from source.entity.scene import Scene

# Keyframe Extraction
from source.preprocess.keyframe_extraction.scene_detector import SceneDetector
from source.preprocess.keyframe_extraction.candidate_extractor import CandidateExtractor
from source.preprocess.keyframe_extraction.keyframe_extractor import KeyframeExtractor
from source.preprocess.feature_extraction.semantic_extractor import SemanticExtractor

# Storage
from source.storage.media.local_media_storage import LocalMediaStorage
from source.storage.metadata.local_metadata_storage import LocalMetadataStorage
from source.storage.vector.local_vector_storage import LocalVectorStorage
from source.storage.video_loader import VideoLoader

# Model 
from source.models.TransNetV2.inference import TransNetV2
from source.models.CLIP.clip import CLIP
from source.models.Embedding.siglip2 import SigLIP2

# Stuff
import time

class PreprocessingPipeline:

    def __init__(self):
        self.scene_detector = SceneDetector(model = TransNetV2(), threshold = 0.75)
        self.candidate_extractor = CandidateExtractor(blur_threshold = 110, duplicate_threshold = 0.7)
        self.keyframe_extractor = KeyframeExtractor(clip_model = CLIP(), similarity_threshold = 0.83, max_frames_per_segment = 6)
        self.feature_extractor = SemanticExtractor(embedding_model=SigLIP2(), batch_size = 8)
        self.media_storage = LocalMediaStorage()
        self.metadata_storage = LocalMetadataStorage()
        self.vector_storage = LocalVectorStorage()

    def run(self, video_path):
        '''
        video = Video(video_path=video_path)

        start = time.perf_counter()
        self.scene_detector.detect(video)
        print(f"[TIME] Scene Detection: {time.perf_counter() - start:.2f}s")
        
        start = time.perf_counter()
        self.candidate_extractor.extract(video)
        print(f"[TIME] Candidate Extraction: {time.perf_counter() - start:.2f}s")

        start = time.perf_counter()
        self.keyframe_extractor.extract(video)
        print(f"[TIME] Keyframe Extraction: {time.perf_counter() - start:.2f}s")

        start = time.perf_counter()
        self.media_storage.save(video)
        output_path = self.metadata_storage.save(video)
        print(f"[TIME] Media Storage: {time.perf_counter() - start:.2f}s")
        '''

        video_loader = VideoLoader()
        video = video_loader.load("data/metadata/L21_V001.json")

        start = time.perf_counter()
        self.feature_extractor.extract(video)
        self.vector_storage.save(video)
        print(f"[TIME] Keyframe Extraction: {time.perf_counter() - start:.2f}s")

        return video