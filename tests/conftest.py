"""Suite-wide isolation of process-global torch numerics.

``seed_everything`` deliberately enables TF32 matmuls for training throughput.
That setting is process-global, so restore it after every test: otherwise a
training test silently lowers the precision of later CUDA parity references.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_matmul_precision():
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
