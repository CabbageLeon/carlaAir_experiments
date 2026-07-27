# ============================================================
#  CarlaAir v0.1.7 — Docker Image
# ============================================================
#  Minimal image. Everything else is done by the deploy script
#  inside the container on first run.
# ============================================================

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps: Unreal Engine runtime requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    vulkan-tools \
    libsdl2-2.0-0 \
    libomp5 \
    xdg-user-dirs \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    unzip \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# uv — Python package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY docker/entrypoint.sh /entrypoint.sh
COPY deploy_carlaair.sh /opt/deploy_carlaair.sh
RUN chmod +x /entrypoint.sh /opt/deploy_carlaair.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["Town10HD"]
