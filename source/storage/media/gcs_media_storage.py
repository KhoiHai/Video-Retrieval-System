from source.storage.media.media_storage import MediaStorage
from source.entity.video import Video

class GCSMediaStorage(MediaStorage):

    def save(self, video: Video):
        raise NotImplementedError