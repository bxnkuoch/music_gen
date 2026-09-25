# Model research (Phase 0, September 2026)

Why we picked these models. Every entry was checked against the model card or repo; the measured numbers are in [results/phase0_benchmark.md](results/phase0_benchmark.md).

**Hardware constraint:** Apple M3, 24 GB unified memory, no NVIDIA GPU. That rules out anything that needs 24 GB of CUDA memory.

## Music generation

| Model | Quality | Runs on M3? | License | Verdict |
|---|---|---|---|---|
| **ACE-Step 1.5** turbo 2B | Best open model (on par with Suno v5 on SongEval); vocals in 50+ languages; controls for BPM, key and reference audio | ✅ measured about 9 s per 30 s clip | MIT | **Chosen** |
| Stable Audio 3 Medium (2B) | Strong music and sound effects; lyrics support unclear | ✅ | Stability Community | Fallback |
| Stable Audio Open Small | Clips of 11 s or less, no vocals | ✅ very fast | Stability Community | Too short |
| MusicGen | Older, instrumental, 30 s | Slow on MPS | CC BY-NC | Outclassed |
| HeartMuLa 3B / YuE / YuE2 | Near the frontier for full songs | ❌ needs 24 GB+ of CUDA memory | Apache / CC BY-NC | Cloud only |

Why ACE-Step: it's permissive, local and the highest quality. It also exposes **explicit controls** (BPM, key, seed, reference audio) that the personalization system can steer, which a black-box API doesn't.

## Audio / music embeddings

| Model | What it gives | License | Verdict |
|---|---|---|---|
| **CLAP** (`larger_clap_music_and_speech`) | Shared text-audio space → zero-shot tags, prompt matching | Apache-2.0 | **Default** |
| **MuQ-MuLan-large** | Shared text-audio space; strong on music-specific retrieval | CC BY-NC (weights) | **Comparison + simulated users** |
| MuQ / MERT | Audio only; strong for training small probes | CC BY-NC | Not needed yet |
| CLaMP 3 | Multimodal music retrieval (uses MERT internally) | MIT | Overkill |
| Essentia (Discogs-EffNet) | 400 interpretable style tags | CC BY-NC-ND; hard to install on Apple Silicon | Skipped |
| librosa | Tempo, loudness, brightness | ISC | **Used** (Phase 2) |

⚠️ `laion/larger_clap_music` (the checkpoint most tutorials recommend) is **broken**: every text embeds to the same vector. See the benchmark report.

## Sources
- ACE-Step 1.5: [GitHub](https://github.com/ace-step/ACE-Step-1.5) · [paper](https://arxiv.org/abs/2602.00744) · [diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/ace_step)
- [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium) · [Stable Audio Open Small](https://huggingface.co/stabilityai/stable-audio-open-small)
- [HeartMuLa](https://github.com/HeartMuLa/heartlib) · [YuE](https://github.com/multimodal-art-projection/YuE) · [MusicGen](https://huggingface.co/docs/transformers/model_doc/musicgen)
- [CLAP](https://huggingface.co/laion/larger_clap_music_and_speech) · [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) · [MuQ paper](https://arxiv.org/abs/2501.01108) · [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) · [CLaMP 3](https://huggingface.co/sander-wood/clamp3)
- [Probing CLAP embeddings (2026)](https://arxiv.org/abs/2607.03806)
