"""Records ground-truth MuQ-MuLan outputs for tests/test_models_slow.py.

Must run under transformers 4.x with the UNPATCHED muq model, so the reference is
independent of our transformers-5 compatibility patch:

    uv run --with "transformers>=4.57,<5" python scripts/record_muq_reference.py
"""

import sys

import numpy as np
import torch
import transformers
from muq import MuQMuLan

from music_gen import audio

sys.path.insert(0, "tests")
from synthetic import chord_with_noise  # noqa: E402

assert transformers.__version__.startswith("4."), transformers.__version__

model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").eval()
x = audio.prepare_mono(chord_with_noise(12.0, 48_000), 48_000, 24_000)
with torch.inference_mode():
    audio_vec = model(wavs=torch.from_numpy(x).unsqueeze(0))[0].numpy()
    text_vec = model(texts=["a calm piano chord"])[0].numpy()
np.savez(
    "tests/fixtures/muq_mulan_reference.npz",
    audio=audio_vec,
    text=text_vec,
    transformers=transformers.__version__,
)
print("saved", audio_vec.shape, text_vec.shape)
