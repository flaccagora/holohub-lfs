import cv2
import argparse
import os
import sys

def extract_frames(video_path, output_dir):
    """
    Extracts frames from a video file and saves them as PNGs formatted for the surgical_scene_recon application.
    The required format is frame-XXXXXX.color.png.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        sys.exit(1)

    frame_idx = 0
    print(f"Extracting frames from '{video_path}' to '{output_dir}'...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # surgical_scene_recon expects files to be named like "frame-000000.color.png"
        frame_filename = os.path.join(output_dir, f"frame-{frame_idx:06d}.color.png")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Success! Extracted {frame_idx} frames to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video for surgical_scene_recon input.")
    parser.add_argument("video_path", help="Path to the input video file (e.g., sample_video.mp4)")
    
    # Default path aligns with the repo structure if running the script from within applications/surgical_scene_recon/
    default_output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "surgical_scene_recon", "frames")
    default_output_dir = os.path.normpath(default_output_dir)
    
    parser.add_argument("--output_dir", "-o", default=default_output_dir, 
                        help=f"Directory to save the extracted frames (default: {default_output_dir})")

    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir)
