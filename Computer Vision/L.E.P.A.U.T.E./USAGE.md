# Usage Guide

## 1. Overview

The L.E.P.A.U.T.E. Framework provides a monocular SE(3) perception pipeline for real-time camera ego-motion estimation, object classification, and relative pose tracking. All entry points are invoked directly through Python. Configuration is managed via `LepauteConfig`, optional external JSON files for camera intrinsics and object scale priors, and command-line arguments.

Core entry points:
- `main.py` — online perception pipeline
- `train.py` — offline training of the SE(3) Residual Refiner
- `convert_bop_to_lepaute.py` — BOP dataset conversion
- Programmatic API via `run_pipeline` and `LepauteConfig`

## 2. Install Requirements

Install dependencies from the provided requirements file:

```bash
pip install -r requirements.txt
```

Ensure a compatible PyTorch installation matching your hardware (CUDA, MPS, or CPU). YOLO weights (`yolov8n.pt`) are downloaded automatically on first use by Ultralytics.

## Dataset Preparation

A pre-converted LEPAUTE dataset is available at:

[https://huggingface.co/datasets/dev1virtuoso/lepaute-dataset](dev1virtuoso/lepaute-dataset)

Alternatively, prepare the dataset from the original BOP source:

1. Download the YCB-V dataset from the Hugging Face BOP benchmark:

```bash
python ycb-v_download.py
```

2. Extract the downloaded `.zip` files located under `./bop_datasets/ycbv`.

3. Convert the extracted data into the monocular LEPAUTE format (computes exact SE(3) relative poses):

```bash
python convert_bop_to_lepaute.py \
  --bop_dir ./dataset/bop_datasets/ycbv \
  --output_dir ./dataset/lepaute_dataset \
  --splits train_pbr train_real \
  --stride 1 \
  --workers 8
```

### Conversion Arguments

| Argument | Default | Description |
|---|---|---|
| `--bop_dir` | Required | Path to the downloaded BOP dataset root |
| `--output_dir` | `./lepaute_dataset` | Destination for converted assets |
| `--splits` | None | Space-separated splits to process (e.g. `train_pbr train_real`) |
| `--stride` | `1` | Frame interval for relative-pose pair generation |
| `--obj_ids` | None | Optional whitespace-separated object IDs to keep |
| `--workers` | `4` | Parallel worker count |
| `--scale` | `1000.0` | Translation normalization factor |

## Model Training

After obtaining or converting the dataset, train the SE(3) Residual Refiner:

```bash
python train.py \
  --dataset_dir ./dataset/lepaute_dataset \
  --checkpoint_dir ./checkpoints \
  --epochs 50 \
  --seed 42
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | Required | Root containing train/test manifests and images |
| `--epochs` | `15` | Maximum training epochs |
| `--checkpoint_dir` | `./checkpoints` | Directory for weight checkpoints |
| `--device` | None | Force device (`cuda`, `mps`, or `cpu`) |
| `--no_compile` | False | Disable `torch.compile` |
| `--seed` | `42` | Random seed for reproducibility |
| `--resume_mode` | `ask` | `resume`, `scratch`, or `ask` |

The trainer automatically detects explicit `train/` and `test/` subdirectories or falls back to random splitting. Checkpoints are written as `latest_checkpoint.pth` and `best_model.pth`.

## Running the Main Pipeline

Launch the online perception system:

```bash
# Standard GUI
python main.py --mode gui --perf medium

# Detailed HUD + trajectory map
python main.py --mode detailedgui --perf high

# Headless (no display)
python main.py --mode headless --perf low

# Structured metric output only
python main.py --mode json
```

### Main Pipeline Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `gui` | `headless`, `gui`, `json`, or `detailedgui` |
| `--perf` | `medium` | `low`, `medium`, or `high` (controls pyramid levels, Gauss-Newton iterations, and frame throttling) |
| `--db` | None | Custom SQLite path for transition storage |
| `--limit` | False | Cap execution at 50 frames (testing) |
| `--log_level` | `general` | `general` (INFO) or `detailed` (DEBUG) |
| `--no_save` | False | Disable SQLite persistence |

## Programmatic Integration

Bypass the CLI and embed the pipeline directly:

```python
from pipeline_and_config import LepauteConfig, DisplayMode
from main import run_pipeline

custom_config = LepauteConfig(
    device="cuda:0",
    fx=600.0,
    fy=600.0,
    cx=320.0,
    cy=240.0,
    object_names=["industrial_arm", "conveyor_belt", "target_widget"]
)

telemetry_results = run_pipeline(
    config=custom_config,
    display_mode=DisplayMode.HEADLESS,
    unlimited=False,
    save_json=False
)

for payload in telemetry_results:
    print(f"Frame {payload['frame_id']} | Object: {payload['category']} | Pose: {payload['xi']}")
```

To inject custom video sources (ROS topics, RTSP, simulation buffers), subclass `CameraIOStream` and override `read()`. Frames must be returned as `(H, W, 3)` uint8 RGB arrays together with a metadata dictionary containing at least `timestamp` and `frame_id`.

## Parameters That Must Be Set Manually

Several values are environment- or hardware-specific and should be configured before production use.

### Camera Intrinsics

Defaults are loaded from `camera_config.json` (or environment variable `LEPAUTE_CAMERA_CONFIG_PATH`). If the file is absent, the following fallbacks are used:

```json
{
  "fx": 250.0,
  "fy": 250.0,
  "cx": 160.0,
  "cy": 120.0
}
```

Override at construction time:

```python
config = LepauteConfig(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
```

Incorrect intrinsics produce scale and pose drift.

### Object Scale Priors

Metric scale priors (meters) are loaded from `object_config.json` (or `LEPAUTE_OBJECT_CONFIG_PATH`). Default keys include:

```json
{
  "table": 1.5,
  "cup": 0.1,
  "keyboard": 0.4,
  "laptop": 0.35,
  "mouse": 0.12,
  "human": 1.7,
  "background": 2.0
}
```

Add or edit entries to match the objects present in your scene. The classifier label is mapped directly to these priors for monocular depth projection.

### Compute Device

Automatically detected (`cuda` > `mps` > `cpu`). Force a specific device via:

```python
config = LepauteConfig(device="cuda")
```

or the CLI flag `--device`.

### Performance Profile

`--perf low|medium|high` adjusts pyramid levels, Gauss-Newton iteration count, and optional frame-rate throttling. Choose according to available compute and required latency.

### Model Checkpoint Path

The inference worker loads `./checkpoints/best_model.pth` by default. Ensure a trained checkpoint exists at this location, or the refiner falls back to random initialization.

### SQLite Database Path

Defaults to `lepaute_data.db`. Override with `--db` or `LepauteConfig(data_store=...)`.

## 3. Troubleshooting

**Thread blocking in asynchronous architectures**  
Avoid high-frequency blocking polls of `get_latest_resolved_state()` inside `asyncio` or ROS spin loops. Pull the latest resolved state asynchronously relative to the worker queue to prevent stream starvation or UI lock-up.

**VRAM leaks in custom execution loops**  
When building loops outside `main.py`, detach intermediate SE(3) tensors produced by `se3_exp_map` and `se3_log_map` (`.detach().cpu().numpy()`) before storing or publishing them. Failure to detach retains the full computational graph and rapidly exhausts GPU memory.

**Concurrency collisions on Apple Silicon (MPS)**  
Heavy main-thread work concurrent with the isolated `InferenceWorker` process can trigger Metal command-buffer crashes. Prefer the built-in CPU fallback, or wrap custom inference blocks with the provided `_mps_lock` / `mps_safe` context manager when forcing MPS execution.

**Camera acquisition failures**  
The capture thread attempts multiple platform-specific backends (AVFoundation, DSHOW, MSMF, V4L2). If connection repeatedly fails, verify device permissions and that no other process holds exclusive access to the camera. Mock mode (`mock=True`) can be used for offline testing.

**Empty or stalled job queue**  
Under sustained high load the inference worker silently drops frames when the bounded queue (size 5) is full. Reduce input frame rate or lower the performance profile if drop counts become excessive.

**Checkpoint loading shape mismatches**  
When resuming training or loading a refined model, the `load_compiled_state_dict` method strips `_orig_mod.` prefixes and tolerates shape mismatches by falling back to fresh initialization for incompatible layers. Verify that the checkpoint was produced with the same `feature_dim` and `max_resolution` settings.
