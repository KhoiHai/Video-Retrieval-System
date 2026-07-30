from source.preprocess.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline("/kaggle/working/keyframes")

video = pipeline.run(
    "/kaggle/input/datasets/whaleeatu/pov-video/Video_01.mp4"
)

'''
for scene in video.scenes:
    print(scene)
'''