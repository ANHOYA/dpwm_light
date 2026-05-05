# Diffusion Policy Light: Can Task

가벼운 `PickPlaceCan` 전용 Diffusion Policy workspace입니다. 원본 `diffusion_policy-main`의 Hydra/workspace/config 구조를 제거하고, Can task 연구에 실제 필요한 최소 파일만 남겼습니다.

This is a lightweight `PickPlaceCan` workspace. It removes the heavy Hydra/workspace/config stack from the original Diffusion Policy repository and keeps only the code needed for Can task teleoperation, dataset preparation, image-policy training, and evaluation.

## Files

```text
can_teleop.py                    # SpaceMouse teleop + demo collection
can_data.py                      # HDF5 dataset preparation + CanImageDataset
can_policy.py                    # ResNet image encoder + ConditionalUnet1D
train_can_image.py               # image diffusion policy training
eval_can_image.py                # robosuite rollout evaluation
diffusion_policy_vision_pusht_demo.ipynb  # original Colab reference notebook
environment.yaml                 # minimal conda environment
```

## Install

```bash
conda env create -f environment.yaml
conda activate robodiff-light
```

Ubuntu packages:

```bash
sudo apt update
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

SpaceMouse packages:

```bash
sudo apt install -y libspnav-dev spacenavd
sudo systemctl start spacenavd
```

GPU check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

## Teleoperation

EEF orientation is locked downward by default. This preserves the custom setting used during Can teleoperation. Use `--free-rotation` only when you intentionally want free rotation.

기본값은 EEF가 아래를 향하도록 orientation lock이 걸려 있습니다. 기존에 수정했던 SpaceMouse 조작감과 EEF 아래 방향 고정이 여기에 포함되어 있습니다.

Manual teleop only:

```bash
python can_teleop.py --mode teleop
```

Collect demonstrations:

```bash
python can_teleop.py --mode collect
```

Start fresh and archive existing HDF5 files:

```bash
python can_teleop.py --mode collect --overwrite
```

Controls:

```text
left SpaceMouse button  : close gripper
right SpaceMouse button : reset/discard episode
s                       : save current episode
r                       : reset/discard episode
q                       : quit
```

Raw demo output:

```text
data/robomimic/datasets/can/custom/demo.hdf5
```

## Prepare Image Dataset

```bash
python can_data.py prepare-image \
  --raw-demo data/robomimic/datasets/can/custom/demo.hdf5 \
  --output data/robomimic/datasets/can/custom/image.hdf5
```

The generated dataset contains:

```text
actions
obs/agentview_image
obs/robot0_eye_in_hand_image
obs/robot0_eef_pos
obs/robot0_eef_quat
obs/robot0_gripper_qpos
next_obs/...
```

## Train Image Diffusion Policy

```bash
python train_can_image.py \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --output data/outputs/can_image_light \
  --device cuda:0
```

Outputs:

```text
data/outputs/can_image_light/latest.pt
data/outputs/can_image_light/best.pt
```

For a quick smoke test:

```bash
python train_can_image.py \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --output data/outputs/can_image_light_debug \
  --num-epochs 1 \
  --max-train-steps 1 \
  --batch-size 2 \
  --num-workers 0 \
  --device cuda:0
```

## Evaluate

```bash
python eval_can_image.py \
  --checkpoint data/outputs/can_image_light/latest.pt \
  --output data/eval_can_image_light \
  --device cuda:0 \
  --num-episodes 4
```

Outputs:

```text
data/eval_can_image_light/eval_log.json
data/eval_can_image_light/episode_*.mp4
```

## Data Policy

Do not commit datasets, checkpoints, logs, or videos.

```text
data/
*.hdf5
*.pt
*.ckpt
*.mp4
```

## Design Notes

- No Hydra.
- No workspace classes.
- No robomimic replay buffer wrapper.
- No zarr cache.
- No kitchen/blockpush/pusht dependencies in the light code path.
- The original notebook is kept only as a reference.
- The maintainable path is the small Python script set above.
