from source.preprocess.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()

video = pipeline.run(
    "data/video/Video_01.mp4"
)

for scene in video.scenes:
    print(scene)