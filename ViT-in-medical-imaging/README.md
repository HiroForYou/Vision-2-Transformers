## Self-attention video generation

You can generate videos like the one on the blog post with `video_generation.py`.

https://user-images.githubusercontent.com/46140458/116817761-47885e80-ab68-11eb-9975-d61d5a919e13.mp4

Extract frames from input video and generate attention video:

```
python video_generation.py  --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path input/video.mp4 \
    --output_path output/ \
    --fps 25
```

Use folder of frames already extracted and generate attention video:

```
python video_generation.py  --pretrained_weights dino_deitsmall8_pretrain.pth \
    --input_path output/frames/ \
    --output_path output/ \
    --resize 256 \
```

Only generate video from folder of attention maps images:

```
python video_generation.py --input_path output/attention \
    --output_path output/ \
    --video_only \
    --video_format avi
```
