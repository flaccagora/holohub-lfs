# Batch Processing Surgical Videos

This document describes how to run `endoscopy_out_of_body_detection` over a folder
of surgical MP4 videos, save per-frame in-body/out-of-body probabilities, and
create videos with out-of-body frames removed.

## Input Data

By default, the batch pipeline reads videos from:

```bash
data/endoscopy_out_of_body_detection/sample_videos
```

The expected layout is one or more videos under that directory, including nested
case folders such as:

```bash
data/endoscopy_out_of_body_detection/sample_videos/case122/case122.mp4
```

The model file is expected in:

```bash
data/endoscopy_out_of_body_detection/out_of_body_detection.onnx
```

## Run the Full Batch Pipeline

Run inside an environment with Holoscan available, typically the Holohub
container:

```bash
cd /workspace/holohub/applications/endoscopy_out_of_body_detection/python
python process_sample_videos.py
```

For each MP4, the script:

1. Decodes the original video with `ffmpeg`.
2. Resizes frames to `256x256`.
3. Converts frames to temporary GXF replayer input.
4. Runs the `endoscopy_out_of_body_detection` Holoscan app.
5. Saves a per-frame probability CSV.
6. Writes a filtered video with out-of-body frames removed.

Temporary GXF files are created in a system temp directory and removed after each
video is processed.

## Outputs

Per-video probability CSV files are written to:

```bash
data/endoscopy_out_of_body_detection/pipeline_output/probabilities/<case>/<case>.csv
```

The combined CSV for all videos is written to:

```bash
data/endoscopy_out_of_body_detection/pipeline_output/all_probabilities.csv
```

Processed videos are written to:

```bash
data/endoscopy_out_of_body_detection/processed_video/<case>/<case>.mp4
```

Each per-frame probability CSV has this format:

```csv
frame,time_seconds,time_hhmmss,in_body_probability,out_of_body_probability,prediction
0,0.000000,00:00:00.000,0.97243210,0.02756790,in-body
1,0.016667,00:00:00.017,0.10481233,0.89518767,out-of-body
```

Frames with `out_of_body_probability >= 0.5` are removed by default.

## Common Options

Use a different input folder:

```bash
python process_sample_videos.py \
  --input-dir /path/to/sample_videos
```

Use a different output folder for probability CSVs:

```bash
python process_sample_videos.py \
  --output-dir /path/to/pipeline_output
```

Use a different folder for processed videos:

```bash
python process_sample_videos.py \
  --processed-video-dir /path/to/processed_video
```

Change the out-of-body removal threshold:

```bash
python process_sample_videos.py \
  --threshold 0.7
```

Only write probability CSVs, without creating processed videos:

```bash
python process_sample_videos.py \
  --no-processed-videos
```

By default the scripts try FFmpeg hardware paths first. The GXF conversion step
tries CUDA/NVDEC decode plus CUDA resize. The processed-video step tries CUDA
decode plus NVENC encode, then falls back through partially accelerated paths
before using CPU-only FFmpeg. To skip CUDA decode and CUDA resize attempts:

```bash
python process_sample_videos.py \
  --no-cuda-ffmpeg
```

To skip NVENC encode attempts:

```bash
python process_sample_videos.py \
  --no-nvenc
```

Processed-video writing uses FFmpeg `select` filtering by default. This keeps
frame-accurate removal inside FFmpeg instead of piping every decoded frame
through Python. The script only uses stream copy for the exact case where every
frame is kept; arbitrary per-frame removal generally requires re-encoding because
compressed video frames depend on surrounding frames and cuts are not guaranteed
to align with keyframes. When re-encoding is required, the script preserves the
source codec family, pixel format, frame rate, resolution, and targets the source
video bitrate so the processed video does not use higher-quality/larger encoder
defaults. To force the older Python frame-pipe path:

```bash
python process_sample_videos.py \
  --pipe-frames
```

## Filter Videos From Existing CSVs

If probability CSVs already exist, rerun only the video filtering step:

```bash
python remove_out_of_body_frames.py \
  --input-dir ../../../data/endoscopy_out_of_body_detection/sample_videos \
  --probability-dir ../../../data/endoscopy_out_of_body_detection/pipeline_output/probabilities \
  --output-dir ../../../data/endoscopy_out_of_body_detection/processed_video
```

Use `--threshold` to control which frames are removed:

```bash
python remove_out_of_body_frames.py \
  --threshold 0.7
```

Use `--no-cuda-ffmpeg` here as well to skip the CUDA decode attempt. Use
`--no-nvenc` to skip NVENC encode attempts.

Use `--pipe-frames` to bypass FFmpeg `select` filtering and use the slower
Python frame-pipe implementation.

## Running One Converted GXF Video Manually

`main.py` can write the probability CSV directly when given a GXF replayer
directory and basename:

```bash
python main.py \
  --config ../endoscopy_out_of_body_detection.yaml \
  --data ../../../data/endoscopy_out_of_body_detection \
  --source replayer \
  --video-dir <gxf_video_dir> \
  --video-basename <gxf_basename> \
  --output-csv <output.csv>
```

`--video-dir` must contain `<gxf_basename>.gxf_entities` and
`<gxf_basename>.gxf_index`.

## Notes

- The full batch pipeline requires the Holoscan Python runtime and GPU inference
  support.
- `ffmpeg` and `ffprobe` are used for MP4 decoding, probing, and writing the
  processed videos. The app-local Dockerfile installs `ffmpeg`.
- FFmpeg hardware acceleration is opportunistic. CUDA/NVDEC decode, CUDA resize,
  and NVENC encode require an FFmpeg build with those capabilities and compatible
  input/output codecs; otherwise the scripts fall back automatically.
- Stream copy is only exact when no frames need to be removed. For frame-level
  removal, the processed video is re-encoded with source-matched settings.
- The processed videos preserve the original video resolution and frame rate, but
  their duration is shorter when out-of-body frames are removed.
- The existing Holohub workflow requirement still applies before running the app
  through the CLI: inspect commands first with `--dryrun --verbose`.
