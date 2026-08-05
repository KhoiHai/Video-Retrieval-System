from source.preprocess.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()

video = pipeline.run(
    "data/video/L21_V001.mkv"
)