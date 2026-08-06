import json

from source.entity.video import Video
from source.entity.scene import Scene
from source.entity.frame import Frame


class VideoLoader:

    def load(self, metadata_path: str) -> Video:

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Create video
        video = Video(
            video_path=metadata["video_path"]
        )

        # Video metadata
        video.fps = metadata["fps"]
        video.frame_count = metadata["frame_count"]
        video.width = metadata["width"]
        video.height = metadata["height"]
        video.duration = metadata["duration"]

        video.clear_scenes()

        # Build scenes
        for scene_metadata in metadata["scenes"]:

            scene = Scene(
                scene_id=scene_metadata["scene_id"],
                start_frame_idx=scene_metadata["start_frame_idx"],
                end_frame_idx=scene_metadata["end_frame_idx"]
            )

            scene.clear_frames()

            # Build frames
            for frame_metadata in scene_metadata["frames"]:

                frame = Frame(
                    frame_idx=frame_metadata["frame_idx"]
                )

                frame.image_uri = frame_metadata["image_uri"]

                scene.add_frame(frame)

            video.add_scene(scene)

        return video