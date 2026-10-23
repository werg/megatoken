"""Optional, independent output-side speech verifier; absent from model training."""

from __future__ import annotations

import math

import numpy as np


class IndependentWhisper:
    def __init__(self, *, model="openai/whisper-base", revision=None, device="cpu"):
        import torch
        import transformers
        from huggingface_hub import HfApi
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.revision = revision or HfApi().model_info(model).sha
        if not self.revision or len(self.revision) != 40:
            raise ValueError("Independent recognition requires an immutable revision")
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model, revision=self.revision)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(model, revision=self.revision)
            .to(device)
            .eval()
        )
        self.metadata = {
            "repository": model,
            "revision": self.revision,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "input_clock": "declared actual waveform rate, scipy polyphase resampled to 16000 Hz",
            "prompt": "language and transcription task only; no target words",
            "maximum_seconds": 30,
        }

    def transcribe(self, waveform, *, sample_rate, language="en"):
        import torch
        from scipy.signal import resample_poly

        samples = np.asarray(waveform, dtype=np.float32)
        if samples.ndim == 2 and samples.shape[0] == 1:
            samples = samples[0]
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("Recognition requires nonempty finite mono decoded audio")
        if type(sample_rate) is not int or sample_rate <= 0 or len(samples) > sample_rate * 30:
            raise ValueError(
                "Recognition needs a positive sample clock and explicit chunking above 30 seconds"
            )
        divisor = math.gcd(sample_rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor).astype(
            np.float32
        )
        features = self.processor(
            samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            generated = self.model.generate(
                features.input_features.to(self.device),
                attention_mask=features.attention_mask.to(self.device),
                language=language,
                task="transcribe",
                max_new_tokens=256,
            )
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
