from source.storage.metadata.metadata_storage import MetadataStorage
from source.entity.video import Video

import os
import json

class LocalMetadataStorage(MetadataStorage):

    def __init__(self, root_dir = "data/metadata"):
        self.root_dir = root_dir

    def save(self, video: Video):
        os.makedirs(self.root_dir, exist_ok=True)

        output_path = os.path.join(
            self.root_dir,
            f"{video.video_id}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                video.to_dict(),
                f,
                indent=4,
                ensure_ascii=False
            )

        return output_path