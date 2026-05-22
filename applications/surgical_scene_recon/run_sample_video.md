# Running Sample Video Reconstruction

This workflow converts `sample_video.mp4` into PNG frames, then runs the
`surgical_scene_recon` pipeline on those frames.

Run all commands from the HoloHub repo root.

## 1. Extract Frames

```bash
python3 applications/surgical_scene_recon/extract_frames.py --overwrite
```

This reads:

```text
sample_video.mp4
```

and writes frames to:

```text
data/surgical_scene_recon/frames/
```

with names like:

```text
frame-000000.color.png
frame-000001.color.png
```

## 2. Preview the Reconstruction Command

Per HoloHub CLI practice, inspect the command before running it:

```bash
./holohub run surgical_scene_recon full --verbose --dryrun
```

## 3. Run Scene Reconstruction

```bash
./holohub run surgical_scene_recon full
```

The full mode runs depth estimation, segmentation, camera pose estimation,
EndoNeRF-format assembly, GSplat training, and the render viewer.

For a headless training-only run:

```bash
./holohub run surgical_scene_recon train --run-args="--headless"
```

Outputs are written under the app build output directory, including:

```text
output/phase1_raw/
output/phase2_vggt/
output/phase3_endonerf/
output/phase4_training/
```
