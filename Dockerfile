# syntax=docker/dockerfile:1
# NVIDIA's release notes identify CUDA 13.4.1 and PyTorch 2.14.0a0 in this image.
# Override with an operator-validated digest to make production pulls immutable.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace/megatoken
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts
COPY docs ./docs

# Keep the container's /etc/pip/constraint.txt in force. --pre admits NVIDIA's
# installed framework build; the explicit check rejects replacing that build.
RUN python - <<'PY'
import subprocess
import sys
from pathlib import Path
import torch

framework = torch.__version__
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-cache-dir", "--pre", ".[train,data,dev,storage]"
], check=True)
installed = subprocess.check_output([
    sys.executable, "-c", "import torch; print(torch.__version__)"
], text=True).strip()
if installed != framework:
    raise RuntimeError(f"Installation replaced the NGC framework: {framework} -> {installed}")
environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
Path("/opt/megatoken-environment.txt").write_text(environment)
PY

# Keep NVIDIA's entrypoint, which configures the installed platform libraries.
CMD ["megatoken", "doctor"]
