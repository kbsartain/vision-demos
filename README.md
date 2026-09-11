# vision-demos

Real-world computer vision demos, rebuilt from scratch.

| Project | What it does | Key model |
|---|---|---|
| [running](running/) | Measures a runner's cadence from a side-on clip, times every foot strike, averages the knee shape at contact, and renders a live analysis panel beside the video. | ViTPose+ Large (local, via Hugging Face transformers) |

Each project has its own README with setup and instructions.

## Origin

This repo started as a replication exercise: rebuild the running-analysis tool
that Jeremy Park demonstrated on LinkedIn, working from the video and the
method described in his [vision-demos](https://github.com/jeremyipark/vision-demos)
repo (Apache-2.0). The code here is a fresh implementation with the same
layout and method, plus local pose backends so it runs without an API key.

Tested on the clip from the original post: 45 strikes at 180.5 spm and a
right knee of 20.8 ± 2.8° at contact, against the 44 strikes, 177 to 182 spm
and 22.7 ± 1.4° shown on the original panel.
