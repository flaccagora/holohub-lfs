"""
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Remove out-of-body frames from surgical videos using per-frame probability CSV files.
"""

import argparse
import csv
import json
import subprocess
from fractions import Fraction
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def keep_frame(row, threshold):
    """Return True when a probability CSV row should be kept in the output video."""
    if "out_of_body_probability" in row and row["out_of_body_probability"] != "":
        return float(row["out_of_body_probability"]) < threshold
    if "prediction" in row and row["prediction"]:
        return row["prediction"].strip().lower() == "in-body"
    if "Out-of-body" in row and row["Out-of-body"] != "":
        return int(float(row["Out-of-body"])) == 0
    raise ValueError("CSV must contain out_of_body_probability, prediction, or Out-of-body")


def read_keep_mask(csv_path, threshold):
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        return [keep_frame(row, threshold) for row in csv.DictReader(csv_file)]


def keep_ranges(keep_mask):
    ranges = []
    start = None
    for index, keep in enumerate(keep_mask):
        if keep and start is None:
            start = index
        elif not keep and start is not None:
            ranges.append((start, index - 1))
            start = None
    if start is not None:
        ranges.append((start, len(keep_mask) - 1))
    return ranges


def output_path_for_video(video_path, input_dir, output_dir):
    relative_path = video_path.relative_to(input_dir)
    return output_dir / relative_path


def probability_csv_for_video(video_path, input_dir, probability_dir):
    relative_path = video_path.relative_to(input_dir)
    return (probability_dir / relative_path).with_suffix(".csv")


def video_metadata(video_path):
    stream = video_stream_info(video_path)
    return stream["width"], stream["height"], stream["fps"]


def video_stream_info(video_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,pix_fmt,bit_rate:format=duration,size,bit_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    stream = metadata["streams"][0]
    fmt = metadata.get("format", {})
    avg_frame_rate = stream.get("avg_frame_rate", "0/0")
    r_frame_rate = stream.get("r_frame_rate", "0/0")
    fps_text = avg_frame_rate if avg_frame_rate != "0/0" else r_frame_rate
    fps = float(Fraction(fps_text)) if fps_text and fps_text != "0/0" else 30.0
    bit_rate = parse_positive_int(stream.get("bit_rate"))
    if bit_rate is None:
        bit_rate = parse_positive_int(fmt.get("bit_rate"))
    if bit_rate is None:
        bit_rate = bitrate_from_size_and_duration(fmt)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "codec_name": stream.get("codec_name", "h264"),
        "pix_fmt": normalized_pix_fmt(stream.get("pix_fmt")),
        "bit_rate": bit_rate,
    }


def parse_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def bitrate_from_size_and_duration(fmt):
    try:
        size_bytes = int(fmt["size"])
        duration_seconds = float(fmt["duration"])
    except (KeyError, TypeError, ValueError):
        return None
    if size_bytes <= 0 or duration_seconds <= 0.0:
        return None
    return int(size_bytes * 8 / duration_seconds)


def normalized_pix_fmt(pix_fmt):
    if not pix_fmt:
        return "yuv420p"
    if pix_fmt == "yuvj420p":
        return "yuv420p"
    return pix_fmt


def ffmpeg_decoder_command(video_path, use_cuda):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
    ]
    if use_cuda:
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    command.extend(["-i", str(video_path), "-map", "0:v:0"])
    if use_cuda:
        command.extend(["-vf", "hwdownload,format=nv12,format=rgb24"])
    command.extend(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    return command


def encoder_name(codec_name, use_nvenc):
    if codec_name in {"h264", "avc1"}:
        return "h264_nvenc" if use_nvenc else "libx264"
    if codec_name in {"hevc", "h265"}:
        return "hevc_nvenc" if use_nvenc else "libx265"
    if use_nvenc:
        raise RuntimeError(f"NVENC does not support source codec '{codec_name}'")
    if codec_name in {"mpeg4", "mjpeg"}:
        return codec_name
    return "libx264"


def bitrate_args(bit_rate):
    if bit_rate is None:
        return []
    buffer_size = max(bit_rate * 2, 1)
    return ["-b:v", str(bit_rate), "-maxrate", str(bit_rate), "-bufsize", str(buffer_size)]


def ffmpeg_encoder_command(output_path, source_info, use_nvenc):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{source_info['width']}x{source_info['height']}",
        "-r",
        str(source_info["fps"]),
        "-i",
        "pipe:0",
        "-an",
    ]
    encoder = encoder_name(source_info["codec_name"], use_nvenc)
    command.extend(["-c:v", encoder])
    if use_nvenc:
        command.extend(["-preset", "p4"])
    command.extend(bitrate_args(source_info["bit_rate"]))
    command.extend(["-pix_fmt", source_info["pix_fmt"], str(output_path)])
    return command


def ffmpeg_stream_copy_command(video_path, output_path):
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0",
        "-c",
        "copy",
        str(output_path),
    ]


def select_expression(ranges):
    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(f"eq(n\\,{start})")
        else:
            parts.append(f"between(n\\,{start}\\,{end})")
    return "+".join(parts)


def ffmpeg_select_command(video_path, output_path, ranges, use_cuda_decode, use_nvenc):
    source_info = video_stream_info(video_path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if use_cuda_decode:
        command.extend(["-hwaccel", "cuda"])
    command.extend(
        [
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            f"select='{select_expression(ranges)}',setpts=N/FRAME_RATE/TB",
            "-r",
            str(source_info["fps"]),
            "-an",
        ]
    )
    encoder = encoder_name(source_info["codec_name"], use_nvenc)
    command.extend(["-c:v", encoder])
    if use_nvenc:
        command.extend(["-preset", "p4"])
    command.extend(bitrate_args(source_info["bit_rate"]))
    command.extend(["-pix_fmt", source_info["pix_fmt"], str(output_path)])
    return command


def summarize_error(error):
    return str(error).splitlines()[0][:300]


def run_ffmpeg(command, action):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed while {action}: {result.stderr}")


def filter_video_with_select(
    video_path,
    csv_path,
    output_path,
    threshold,
    use_cuda_decode,
    use_nvenc,
):
    keep_mask = read_keep_mask(csv_path, threshold)
    ranges = keep_ranges(keep_mask)
    if not ranges:
        raise RuntimeError(f"No in-body frames selected for {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(ranges) == 1 and ranges[0] == (0, len(keep_mask) - 1):
        run_ffmpeg(ffmpeg_stream_copy_command(video_path, output_path), f"copying {video_path}")
    else:
        run_ffmpeg(
            ffmpeg_select_command(
                video_path, output_path, ranges, use_cuda_decode, use_nvenc
            ),
            f"filtering {video_path}",
        )

    kept_frames = sum(keep_mask)
    return {
        "input_frames": len(keep_mask),
        "probability_rows": len(keep_mask),
        "kept_frames": kept_frames,
        "removed_frames": len(keep_mask) - kept_frames,
        "output_path": output_path,
    }


def filter_video_with_commands(
    video_path, csv_path, output_path, threshold, decoder_command, encoder_command
):
    keep_mask = read_keep_mask(csv_path, threshold)
    width, height, fps = video_metadata(video_path)
    frame_size = width * height * 3
    output_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_index = 0
    kept_frames = 0
    try:
        while True:
            frame = decoder.stdout.read(frame_size)
            if not frame:
                break
            if len(frame) != frame_size:
                raise RuntimeError(f"Decoded incomplete frame from {video_path}")
            if frame_index < len(keep_mask) and keep_mask[frame_index]:
                encoder.stdin.write(frame)
                kept_frames += 1
            frame_index += 1
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        decoder_stderr = decoder.stderr.read()
        encoder_stderr = encoder.stderr.read()
        decoder.wait()
        encoder.wait()

    if decoder.returncode:
        raise RuntimeError(f"ffmpeg failed while decoding {video_path}: {decoder_stderr.decode()}")
    if encoder.returncode:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}: {encoder_stderr.decode()}")

    return {
        "input_frames": frame_index,
        "probability_rows": len(keep_mask),
        "kept_frames": kept_frames,
        "removed_frames": max(0, min(frame_index, len(keep_mask)) - kept_frames),
        "output_path": output_path,
    }


def filter_video_with_options(
    video_path, csv_path, output_path, threshold, use_cuda_decode, use_nvenc
):
    source_info = video_stream_info(video_path)
    return filter_video_with_commands(
        video_path,
        csv_path,
        output_path,
        threshold,
        ffmpeg_decoder_command(video_path, use_cuda=use_cuda_decode),
        ffmpeg_encoder_command(output_path, source_info, use_nvenc=use_nvenc),
    )


def filter_video(
    video_path,
    csv_path,
    output_path,
    threshold=0.5,
    use_cuda_ffmpeg=True,
    use_nvenc=True,
    use_select_filter=True,
):
    attempts = []
    if use_cuda_ffmpeg:
        attempts.extend(
            [
                ("CUDA decode + NVENC encode", True, use_nvenc),
                ("CUDA decode + CPU encode", True, False),
            ]
        )
    if use_nvenc:
        attempts.append(("CPU decode + NVENC encode", False, True))
    attempts.append(("CPU decode + CPU encode", False, False))

    tried = set()
    for label, use_cuda_decode, use_nvenc_encode in attempts:
        key = (use_cuda_decode, use_nvenc_encode)
        if key in tried:
            continue
        tried.add(key)
        try:
            if use_select_filter:
                return filter_video_with_select(
                    video_path,
                    csv_path,
                    output_path,
                    threshold,
                    use_cuda_decode,
                    use_nvenc_encode,
                )
            else:
                return filter_video_with_options(
                    video_path,
                    csv_path,
                    output_path,
                    threshold,
                    use_cuda_decode,
                    use_nvenc_encode,
                )
        except RuntimeError as error:
            print(f"  {label} failed, trying next path: {summarize_error(error)}")
            if output_path.exists():
                output_path.unlink()

    if use_select_filter:
        print("  FFmpeg select filter paths failed, falling back to Python frame pipe")
        return filter_video(
            video_path,
            csv_path,
            output_path,
            threshold,
            use_cuda_ffmpeg,
            use_nvenc,
            use_select_filter=False,
        )

    raise RuntimeError(f"Unable to filter {video_path} with any FFmpeg decode/encode path")


def iter_videos(input_dir):
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def filter_video_folder(
    input_dir,
    probability_dir,
    output_dir,
    threshold=0.5,
    use_cuda_ffmpeg=True,
    use_nvenc=True,
    use_select_filter=True,
):
    results = []
    for video_path in iter_videos(input_dir):
        csv_path = probability_csv_for_video(video_path, input_dir, probability_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing probability CSV for {video_path}: {csv_path}")
        output_path = output_path_for_video(video_path, input_dir, output_dir)
        results.append(
            filter_video(
                video_path,
                csv_path,
                output_path,
                threshold,
                use_cuda_ffmpeg,
                use_nvenc,
                use_select_filter,
            )
        )
    return results


def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "data" / "endoscopy_out_of_body_detection"

    parser = argparse.ArgumentParser(
        description="Remove out-of-body frames from original surgical videos."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_dir / "sample_videos",
        help="Folder containing original surgical videos",
    )
    parser.add_argument(
        "--probability-dir",
        type=Path,
        default=data_dir / "pipeline_output" / "probabilities",
        help="Folder containing per-frame probability CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir / "processed_video",
        help="Folder for videos with out-of-body frames removed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Remove frames with out_of_body_probability greater than or equal to this value",
    )
    parser.add_argument(
        "--no-cuda-ffmpeg",
        dest="use_cuda_ffmpeg",
        action="store_false",
        help="Disable FFmpeg CUDA/NVDEC decode attempts",
    )
    parser.add_argument(
        "--no-nvenc",
        dest="use_nvenc",
        action="store_false",
        help="Disable FFmpeg NVENC encode attempts",
    )
    parser.add_argument(
        "--pipe-frames",
        dest="use_select_filter",
        action="store_false",
        help="Use the slower Python frame pipe instead of FFmpeg select filtering",
    )
    parser.set_defaults(use_cuda_ffmpeg=True, use_nvenc=True, use_select_filter=True)
    return parser.parse_args()


def main():
    args = parse_args()
    results = filter_video_folder(
        args.input_dir.resolve(),
        args.probability_dir.resolve(),
        args.output_dir.resolve(),
        args.threshold,
        args.use_cuda_ffmpeg,
        args.use_nvenc,
        args.use_select_filter,
    )
    for result in results:
        print(
            f"{result['output_path']}: kept {result['kept_frames']} / "
            f"{result['input_frames']} frames"
        )


if __name__ == "__main__":
    main()
