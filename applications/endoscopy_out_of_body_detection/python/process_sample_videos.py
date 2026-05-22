"""
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Batch pipeline for endoscopy out-of-body detection on MP4 surgical videos.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remove_out_of_body_frames import (  # noqa: E402
    VIDEO_EXTENSIONS,
    filter_video,
    output_path_for_video,
    probability_csv_for_video,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "endoscopy_out_of_body_detection"
DEFAULT_CONFIG = APP_DIR.parent / "endoscopy_out_of_body_detection.yaml"


def iter_videos(input_dir):
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def video_metadata(video_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    width, height, avg_frame_rate, r_frame_rate = result.stdout.strip().split(",")[:4]
    fps_text = avg_frame_rate if avg_frame_rate != "0/0" else r_frame_rate
    fps = float(Fraction(fps_text)) if fps_text and fps_text != "0/0" else 30.0
    return int(width), int(height), fps


def ffmpeg_decode_command(video_path, width, height, use_cuda):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if use_cuda:
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    command.extend(["-i", str(video_path), "-map", "0:v:0"])
    if use_cuda:
        command.extend(
            [
                "-vf",
                f"scale_cuda={width}:{height},hwdownload,format=nv12,format=rgb24",
            ]
        )
    else:
        command.extend(["-vf", f"scale={width}:{height}"])
    command.extend(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    return command


def remove_gxf_outputs(output_dir, basename):
    for suffix in ("gxf_entities", "gxf_index"):
        path = output_dir / f"{basename}.{suffix}"
        if path.exists():
            path.unlink()


def summarize_error(error):
    return str(error).splitlines()[0][:300]


def convert_video_to_gxf_with_command(video_path, output_dir, basename, width, height, command):
    sys.path.insert(0, str(REPO_ROOT / "utilities"))
    from gxf_entity_codec import EntityWriter

    _, _, fps = video_metadata(video_path)
    frame_count = 0
    frame_size = width * height * 3
    output_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with EntityWriter(directory=str(output_dir), basename=basename, framerate=fps) as writer:
            while True:
                frame_bytes = process.stdout.read(frame_size)
                if not frame_bytes:
                    break
                if len(frame_bytes) != frame_size:
                    raise RuntimeError(f"Decoded incomplete frame from {video_path}")
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
                writer.add(np.ascontiguousarray(frame), name="")
                frame_count += 1
    finally:
        _, stderr = process.communicate()

    if process.returncode:
        raise RuntimeError(f"ffmpeg failed while decoding {video_path}: {stderr.decode()}")

    if frame_count == 0:
        raise RuntimeError(f"No frames were decoded from {video_path}")
    return frame_count


def convert_video_to_gxf(video_path, output_dir, basename, width, height, use_cuda_ffmpeg=True):
    try_cuda = use_cuda_ffmpeg
    if try_cuda:
        try:
            return convert_video_to_gxf_with_command(
                video_path,
                output_dir,
                basename,
                width,
                height,
                ffmpeg_decode_command(video_path, width, height, use_cuda=True),
            )
        except RuntimeError as error:
            print(f"  CUDA ffmpeg decode failed, falling back to CPU: {summarize_error(error)}")
            remove_gxf_outputs(output_dir, basename)

    return convert_video_to_gxf_with_command(
        video_path,
        output_dir,
        basename,
        width,
        height,
        ffmpeg_decode_command(video_path, width, height, use_cuda=False),
    )


def run_detection_app(args, gxf_dir, basename, csv_path, frame_rate):
    command = [
        sys.executable,
        str(APP_DIR / "main.py"),
        "--config",
        str(args.config),
        "--data",
        str(args.data_dir),
        "--source",
        "replayer",
        "--video-dir",
        str(gxf_dir),
        "--video-basename",
        basename,
        "--output-csv",
        str(csv_path),
        "--frame-rate",
        f"{frame_rate:.8f}",
    ]
    env = os.environ.copy()
    env.setdefault("HOLOHUB_DATA_PATH", str(args.data_dir.parent))
    subprocess.run(command, check=True, cwd=str(APP_DIR), env=env)


def append_combined_csv(combined_csv, video_path, input_dir, probability_csv):
    write_header = not combined_csv.exists() or combined_csv.stat().st_size == 0
    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    relative_video = video_path.relative_to(input_dir).as_posix()

    with probability_csv.open(newline="") as input_file, combined_csv.open(
        "a", newline=""
    ) as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = ["video", *reader.fieldnames]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in reader:
            writer.writerow({"video": relative_video, **row})


def process_video(args, video_path, combined_csv):
    probability_csv = probability_csv_for_video(
        video_path, args.input_dir, args.probability_dir
    )
    output_video = output_path_for_video(video_path, args.input_dir, args.processed_video_dir)
    probability_csv.parent.mkdir(parents=True, exist_ok=True)

    relative_video = video_path.relative_to(args.input_dir)
    print(f"Processing {relative_video}")
    with tempfile.TemporaryDirectory(prefix="endoscopy_oob_gxf_") as work_dir:
        basename = video_path.stem
        _, _, frame_rate = video_metadata(video_path)
        frame_count = convert_video_to_gxf(
            video_path,
            Path(work_dir),
            basename,
            args.conversion_width,
            args.conversion_height,
            args.use_cuda_ffmpeg,
        )
        print(f"  converted {frame_count} frames to GXF")
        run_detection_app(args, Path(work_dir), basename, probability_csv, frame_rate)
        print(f"  wrote probabilities to {probability_csv}")

    append_combined_csv(combined_csv, video_path, args.input_dir, probability_csv)
    if args.write_processed_videos:
        result = filter_video(
            video_path,
            probability_csv,
            output_video,
            args.threshold,
            args.use_cuda_ffmpeg,
            args.use_nvenc,
            args.use_select_filter,
        )
        print(
            f"  wrote {result['output_path']} "
            f"({result['kept_frames']} / {result['input_frames']} frames kept)"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run endoscopy_out_of_body_detection on every video in the sample_videos folder "
            "and write per-frame probabilities plus processed videos."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "sample_videos",
        help="Folder containing original surgical videos",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing out_of_body_detection.onnx",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to endoscopy_out_of_body_detection.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "pipeline_output",
        help="Folder for pipeline CSV output",
    )
    parser.add_argument(
        "--processed-video-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "processed_video",
        help="Folder for videos with out-of-body frames removed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Remove frames with out_of_body_probability greater than or equal to this value",
    )
    parser.add_argument(
        "--conversion-width",
        type=int,
        default=256,
        help="Width used when converting MP4 frames to GXF for inference",
    )
    parser.add_argument(
        "--conversion-height",
        type=int,
        default=256,
        help="Height used when converting MP4 frames to GXF for inference",
    )
    parser.add_argument(
        "--no-processed-videos",
        dest="write_processed_videos",
        action="store_false",
        help="Only write probability CSV files",
    )
    parser.add_argument(
        "--no-cuda-ffmpeg",
        dest="use_cuda_ffmpeg",
        action="store_false",
        help="Disable FFmpeg CUDA/NVDEC decode and CUDA resize attempts",
    )
    parser.add_argument(
        "--no-nvenc",
        dest="use_nvenc",
        action="store_false",
        help="Disable FFmpeg NVENC encode attempts when writing processed videos",
    )
    parser.add_argument(
        "--pipe-frames",
        dest="use_select_filter",
        action="store_false",
        help="Use the slower Python frame pipe when writing processed videos",
    )
    parser.set_defaults(
        write_processed_videos=True,
        use_cuda_ffmpeg=True,
        use_nvenc=True,
        use_select_filter=True,
    )
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.probability_dir = args.output_dir / "probabilities"
    args.processed_video_dir = args.processed_video_dir.resolve()
    return args


def main():
    args = parse_args()
    videos = iter_videos(args.input_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in {args.input_dir}")

    combined_csv = args.output_dir / "all_probabilities.csv"
    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_csv.write_text("", encoding="utf-8")

    for video_path in videos:
        process_video(args, video_path, combined_csv)

    print(f"Combined probability CSV: {combined_csv}")
    if args.write_processed_videos:
        print(f"Processed videos: {args.processed_video_dir}")


if __name__ == "__main__":
    main()
