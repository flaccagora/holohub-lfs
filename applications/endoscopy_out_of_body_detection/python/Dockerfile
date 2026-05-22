# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data
