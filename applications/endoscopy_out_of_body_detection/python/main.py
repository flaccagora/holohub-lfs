"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import atexit
import csv
import copy
import os
from pathlib import Path

import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    InferenceOp,
    InferenceProcessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator


class CsvProbabilityWriterOp(Operator):
    """Write per-frame in-body/out-of-body probabilities from inference output."""

    def __init__(
        self,
        *args,
        output_csv,
        frame_rate=0.0,
        in_tensor_name="out_of_body_inferred",
        **kwargs,
    ):
        self.output_csv = Path(output_csv)
        self.frame_rate = float(frame_rate) if frame_rate else 0.0
        self.in_tensor_name = in_tensor_name
        self.frame_index = 0
        self._file = None
        self._writer = None
        super().__init__(*args, **kwargs)
        atexit.register(self.close)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def _open(self):
        if self._file is not None:
            return
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_csv.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "frame",
                "time_seconds",
                "time_hhmmss",
                "in_body_probability",
                "out_of_body_probability",
                "prediction",
            ]
        )

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    @staticmethod
    def _to_numpy(tensor):
        try:
            import cupy as cp

            return cp.asnumpy(cp.asarray(tensor))
        except Exception:
            return np.asarray(tensor)

    @staticmethod
    def _binary_probabilities(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size >= 2:
            scores = values[:2]
            if np.all(scores >= 0.0) and np.all(scores <= 1.0) and np.isclose(
                scores.sum(), 1.0, atol=1e-3
            ):
                probabilities = scores
            else:
                scores = scores - np.max(scores)
                exp_scores = np.exp(scores)
                probabilities = exp_scores / exp_scores.sum()
            return float(probabilities[0]), float(probabilities[1])

        if values.size == 1:
            score = float(values[0])
            out_probability = (
                score if 0.0 <= score <= 1.0 else float(1.0 / (1.0 + np.exp(-score)))
            )
            return 1.0 - out_probability, out_probability

        raise ValueError("Out-of-body inference tensor is empty")

    def _frame_time_seconds(self):
        if self.frame_rate <= 0.0:
            return None
        return self.frame_index / self.frame_rate

    @staticmethod
    def _format_timestamp(time_seconds):
        if time_seconds is None:
            return ""
        whole_seconds = int(time_seconds)
        milliseconds = int(round((time_seconds - whole_seconds) * 1000))
        if milliseconds == 1000:
            whole_seconds += 1
            milliseconds = 0
        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        tensor = in_message[self.in_tensor_name]
        in_probability, out_probability = self._binary_probabilities(self._to_numpy(tensor))
        prediction = "out-of-body" if out_probability > in_probability else "in-body"
        time_seconds = self._frame_time_seconds()

        self._open()
        self._writer.writerow(
            [
                self.frame_index,
                "" if time_seconds is None else f"{time_seconds:.6f}",
                self._format_timestamp(time_seconds),
                f"{in_probability:.8f}",
                f"{out_probability:.8f}",
                prediction,
            ]
        )
        self.frame_index += 1


class EndoscopyOutOfBodyDetectionApp(Application):
    """Endoscopy Out-of-Body Detection"""

    def __init__(
        self,
        data_dir,
        source="replayer",
        do_record=False,
        enable_analytics=False,
        video_dir=None,
        video_basename=None,
        output_csv=None,
        frame_rate=0.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.source = source.lower()
        if self.source not in ["replayer", "aja"]:
            raise ValueError(
                f"Unsupported source: {source}. Please choose either 'replayer' or 'aja'."
            )
        self.do_record = do_record
        self.enable_analytics = enable_analytics
        self.video_dir = video_dir
        self.video_basename = video_basename
        self.output_csv = output_csv
        self.frame_rate = frame_rate

    def compose(self):
        """Compose the Holoscan application pipeline."""

        # Select the input source
        is_aja = self.source == "aja"
        if is_aja:
            from holohub.aja_source import AJASourceOp

            source = AJASourceOp(self, name="aja_source", **self.kwargs("aja"))
        else:
            replayer_config = "analytics_replayer" if self.enable_analytics else "replayer"
            replayer_kwargs = dict(self.kwargs(replayer_config))
            if self.video_basename:
                replayer_kwargs["basename"] = self.video_basename
                replayer_kwargs["repeat"] = False
                replayer_kwargs["realtime"] = False
            video_dir = self.video_dir or self.data_dir
            source = VideoStreamReplayerOp(
                self, name="video_replayer", directory=video_dir, **replayer_kwargs
            )

        # Memory allocator for some operators
        pool = UnboundedAllocator(self, name="pool")
        in_dtype = "rgba8888" if is_aja else "rgb888"

        # Format conversion (ensures correct format for inference)
        out_of_body_preprocessor = FormatConverterOp(
            self,
            name="out_of_body_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("out_of_body_preprocessor"),
        )

        # AI Model Inference (detects whether endoscope is inside or outside the body)
        inference_kwargs = copy.deepcopy(self.kwargs("out_of_body_inference"))
        for k, v in inference_kwargs["model_path_map"].items():
            inference_kwargs["model_path_map"][k] = os.path.join(self.data_dir, v)
        out_of_body_inference = InferenceOp(
            self, name="out_of_body_inference", allocator=pool, **inference_kwargs
        )

        if self.output_csv:
            out_of_body_postprocessor = CsvProbabilityWriterOp(
                self,
                name="out_of_body_probability_writer",
                output_csv=self.output_csv,
                frame_rate=self.frame_rate,
            )
        else:
            postprocess_config = (
                "analytics_out_of_body_postprocessor"
                if self.enable_analytics
                else "out_of_body_postprocessor"
            )
            out_of_body_postprocessor = InferenceProcessorOp(
                self,
                name="out_of_body_postprocessor",
                allocator=pool,
                disable_transmitter=True,
                **self.kwargs(postprocess_config),
            )

        # Define the pipeline connections
        if is_aja:
            self.add_flow(source, out_of_body_preprocessor, {("video_buffer_output", "")})
        else:
            self.add_flow(source, out_of_body_preprocessor)
        self.add_flow(out_of_body_preprocessor, out_of_body_inference, {("", "receivers")})
        self.add_flow(
            out_of_body_inference,
            out_of_body_postprocessor,
            {("transmitter", "in" if self.output_csv else "receivers")},
        )


def main(args):
    app = EndoscopyOutOfBodyDetectionApp(
        args.data,
        args.source,
        args.record,
        args.analytics,
        args.video_dir,
        args.video_basename,
        args.output_csv,
        args.frame_rate,
    )
    app.config(args.config)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Endoscopy Out-of-Body Detection (Python)")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=os.environ.get("HOLOHUB_DATA_PATH", "../data"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja"],
        default="replayer",
        help="Input source (default: replayer)",
    )
    parser.add_argument("--record", action="store_true", help="Record the input video")
    parser.add_argument("--analytics", action="store_true", help="Enable analytics")
    parser.add_argument(
        "--video-dir",
        type=str,
        help="Directory containing the GXF entities and index to replay",
    )
    parser.add_argument(
        "--video-basename",
        type=str,
        help="Basename of the GXF entities and index to replay",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Write per-frame in-body/out-of-body probabilities to this CSV file",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=0.0,
        help="Original video frame rate used to log frame timestamps in --output-csv",
    )
    args = parser.parse_args()
    print("ARGS:", args)
    main(args)
