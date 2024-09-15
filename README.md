Anonymize Face
====

Anonymize faces in videos.


Gaussian blur is a common technique to blur people's faces, however given the appropriate conditions ([0]) **it _can_ be reversed** defeating the purpose of it.

This toolkit uses [deepinsight/insightface][deepinsight/insightface], one of the most accurate DNN Models as of today for facial recognition tasks.

[deepinsight/insightface]: https://github.com/deepinsight/insightface

[0]: https://www.sciencedirect.com/science/article/abs/pii/S0734189X87801536


Usage
-----

Get your `venv` with dependencies:

```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### CUDA ONLY

If you have CUDA-capable card install the cuda requirements with:

```bash
pip install -r requirements-cuda.txt
```

And make sure to have the cuda libs in your `LD_LIBRARY_PATH`:

```bash
eval $(python generate_cuda_ld_path.py)
```

### / CUDA ONLY

Then:

```bash
./anonymize-face.py -h # help

./anonymize-face.py path/to/your/vid.mp4
```

Will generate the following files:

- `path/to/your/vid.overlay.mp4`: a %color% over black overlay
- `path/to/your/vid.bboxes.bin`: a pickled representation of the bounding boxes of the faces found in the video indexed by frame. Like:  
    ```python
    bboxes : Array[Array[frame_num: int, faces: Array[int, int, int, int]]
    [
        [1, [(x1, y1, x2, y2)]], # one frame, one face
        [3, [(x1, y1, x2, y2), (x1, y1, x2, y2)...]], # one frame, many faces
    ]
    ```

Merging the overlay with the original video
---

If audio is not needed, you can use:

```
./anonymize-face.py --write-merged path/to/your/vid.mp4
```

This will generate `path/to/your/vid.merged.mp4` with the overlay generated before blended into the original video.

Else, use `ffmpeg_merger.bash` to preserve the original audio.

```
./anonymize-face.py path/to/your/vid.mp4 # from before
bash ffmpeg_merger.bash path/to/your/vid.mp4
```

Additionally the `ffmpeg_merger.bash` script applies a gaussian blur for after the overlay is applied to maximize blending and covering visible identifying features.

