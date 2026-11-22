# syntax=docker/dockerfile:1
#
# Inkling-2-Mini — Fused Vulkan llama.cpp (vengeful180-dot/llama.cpp) on Intel Arc Pro B70
# Fused loop/depth-mix/router ops + native NVFP4 mat-vec + F16 relative-position bias.
# Serves OpenAI-compatible API on port 8080.
#

FROM ubuntu:26.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential cmake ninja-build pkg-config \
    libvulkan-dev vulkan-tools mesa-vulkan-drivers glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://github.com/vengeful180-dot/llama.cpp.git /app/llama.cpp

RUN cmake -S /app/llama.cpp -B /app/llama.cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF && \
    cmake --build /app/llama.cpp/build \
        --target llama-server llama-bench test-backend-ops \
        -j "$(nproc)"

EXPOSE 8080

ENTRYPOINT ["/app/llama.cpp/build/bin/llama-server"]
