import argparse
import sys
from pathlib import Path

import cv2


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "sample_video.mp4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "surgical_scene_recon" / "frames"


def extract_frames(video_path, output_dir, *, overwrite=False, stride=1, max_frames=None):
    """
    Extract frames from a video file and save them in the format expected by
    surgical_scene_recon: frame-000000.color.png, frame-000001.color.png, ...
    """
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("--max-frames must be >= 1")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file '{video_path}' not found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_dir.glob("*.png"))
    if existing_frames and not overwrite:
        raise FileExistsError(
            f"Output directory '{output_dir}' already contains PNG frames. "
            "Pass --overwrite to replace the existing frame set."
        )
    if overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video '{video_path}'.")

    source_frame_idx = 0
    output_frame_idx = 0
    print(f"Extracting frames from '{video_path}' to '{output_dir}'...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if source_frame_idx % stride == 0:
                frame_filename = output_dir / f"frame-{output_frame_idx:06d}.color.png"
                if not cv2.imwrite(str(frame_filename), frame):
                    raise RuntimeError(f"Failed to write frame '{frame_filename}'.")
                output_frame_idx += 1

                if max_frames is not None and output_frame_idx >= max_frames:
                    break

            source_frame_idx += 1
    finally:
        cap.release()

    if output_frame_idx == 0:
        raise RuntimeError(f"No frames were extracted from '{video_path}'.")

    print(f"Success: extracted {output_frame_idx} frames to '{output_dir}'.")
    print("Next step:")
    print("  ./holohub run surgical_scene_recon full")
    return output_frame_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert an MP4 or other video into PNG frames for the "
            "surgical_scene_recon application."
        )
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=str(DEFAULT_VIDEO_PATH),
        help=f"Path to the input video file (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save extracted frames (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace PNG frames already present in the output directory.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth source frame (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after writing this many output frames.",
    )

    args = parser.parse_args()
    try:
        extract_frames(
            args.video_path,
            args.output_dir,
            overwrite=args.overwrite,
            stride=args.stride,
            max_frames=args.max_frames,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
