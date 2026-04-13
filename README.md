# depth_stream

Realtime Intel RealSense + LingBot-Depth streaming (standalone). Mirrors `new streaming/stream_realsense_lingbot.py` with the repo root renamed to `depth_stream` so it can be pushed as its own project.

## Setup

```bash
conda env create -f environment.yml
conda activate lingbot-depth-stream
```

If the env already exists:

```bash
conda env update -n lingbot-depth-stream -f environment.yml --prune
```

## Run realtime streaming

```bash
python stream_realsense_lingbot.py
```

Common options:

```bash
python stream_realsense_lingbot.py \
  --width 640 --height 480 --fps 60 \
  --resolution-level 6 \
  --infer-every 2 \
  --infer-width 512 --infer-height 384 \
  --min-depth 0.2 --max-depth 5.0
```

If a requested mode is unsupported, the script auto-falls back to the nearest supported RGB-D setup and prints the selected mode. Quick probe (no GUI):

```bash
python stream_realsense_lingbot.py --fps 60 --probe-only
```

## Controls

- `q` or `Esc`: quit
- `s`: save current RGB/depth/model frames to `captures/`

## Expected layout

```
k_depth/
  depth_stream/           # this repo (ready to push)
  lingbot-depth/          # sibling checkout of LingBot-Depth (Apache-2.0)
```

The script dynamically adds `../lingbot-depth` to `PYTHONPATH` at runtime; no submodule is used so you can keep the heavy model repo separate.

## Licensing

The streaming script depends on the LingBot-Depth codebase (Apache-2.0). No additional license files were added here; include the upstream `lingbot-depth/LICENSE` alongside if you vendor code.
