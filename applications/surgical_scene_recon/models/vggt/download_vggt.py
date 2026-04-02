#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Download VGG16 model for offline use on compute nodes.

This script downloads the VGGT-1B model from Hugging Face and saves it locally.
Run this on a login node with network access, then transfer the model to compute nodes.

Usage:
    python download_vggt.py --output-dir /path/to/vggt_models

    # On compute nodes, pass the checkpoint path to vggt_inference.py:
    python vggt_inference.py \\
        --image-dir /path/to/images \\
        --output-dir /path/to/output \\
        --vggt-checkpoint /path/to/vggt_models/pytorch_model.bin
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch


def download_vggt_model(output_dir: str) -> str:
    """Download VGGT-1B model from Hugging Face.

    Parameters
    ----------
    output_dir : str
        Directory to save the model. The model will be saved as pytorch_model.bin
        along with config.json and other metadata files.

    Returns
    -------
    str
        Path to the downloaded model directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[download_vggt] Downloading VGGT-1B model to {output_dir}")
    print("[download_vggt] This may take several minutes...")

    try:
        from transformers import AutoModel

        # Download model using Hugging Face transformers
        # This will download to the huggingface cache first, then we'll copy it
        print("[download_vggt] Fetching VGGT-1B from Hugging Face...")
        model = AutoModel.from_pretrained("facebook/VGGT-1B", trust_remote_code=True)

        # Save model locally
        model_save_path = output_path / "pytorch_model.bin"
        print(f"[download_vggt] Saving model to {model_save_path}")
        torch.save(model.state_dict(), str(model_save_path))

        # Also save the config
        config_path = output_path / "config.json"
        print(f"[download_vggt] Saving config to {config_path}")
        model.config.save_pretrained(str(output_path))

        print(f"[download_vggt] ✓ Model saved successfully to {output_dir}")
        return str(output_path)

    except ImportError:
        print("[download_vggt] ✗ Error: transformers library not found")
        print("[download_vggt]   Install with: pip install transformers")
        sys.exit(1)
    except Exception as e:
        print(f"[download_vggt] ✗ Error downloading model: {e}")
        sys.exit(1)


def verify_model(model_dir: str) -> bool:
    """Verify that the model files are present.

    Parameters
    ----------
    model_dir : str
        Directory containing the model files.

    Returns
    -------
    bool
        True if model files are present and valid.
    """
    model_path = Path(model_dir)
    required_files = ["pytorch_model.bin", "config.json"]

    print(f"\n[download_vggt] Verifying model in {model_dir}")
    all_present = True
    for fname in required_files:
        fpath = model_path / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {fname} (MISSING)")
            all_present = False

    return all_present


def main():
    parser = argparse.ArgumentParser(
        description="Download VGGT-1B model for offline use on compute nodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the VGGT-1B model files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing model without downloading",
    )

    args = parser.parse_args()

    if args.verify_only:
        if verify_model(args.output_dir):
            print("[download_vggt] ✓ Model verification successful")
            sys.exit(0)
        else:
            print("[download_vggt] ✗ Model verification failed")
            sys.exit(1)

    try:
        download_vggt_model(args.output_dir)
        if verify_model(args.output_dir):
            print("\n[download_vggt] ✓ All done! Model is ready for use.")
            print(f"[download_vggt]   Pass to vggt_inference.py: --vggt-checkpoint {args.output_dir}")
            sys.exit(0)
        else:
            print("[download_vggt] ✗ Model verification failed after download")
            sys.exit(1)
    except Exception as e:
        print(f"\n[download_vggt] ✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
