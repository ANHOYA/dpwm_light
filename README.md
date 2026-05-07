# Diffusion Policy Light: Can Task

가벼운 `PickPlaceCan` 전용 Diffusion Policy workspace입니다. 원본 `diffusion_policy-main`의 Hydra/workspace/config 구조를 제거하고, Can task 연구에 실제 필요한 최소 파일만 남겼습니다.

This is a lightweight `PickPlaceCan` workspace. It removes the heavy Hydra/workspace/config stack from the original Diffusion Policy repository and keeps only the code needed for Can task teleoperation, dataset preparation, image-policy training, and evaluation.

## Files

```text
can_teleop.py                    # SpaceMouse teleop + demo collection
can_data.py                      # HDF5 dataset preparation + CanImageDataset
can_policy.py                    # ResNet image encoder + ConditionalUnet1D
can_image_world_model.py         # multi-frame future image latent world model
train_can_image.py               # image diffusion policy training
train_image_world_model.py       # image world model training
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

GPU compatibility note:

```text
This lightweight environment intentionally uses the old robodiff-compatible
PyTorch 1.12.1 + CUDA 11.6 stack. It works with RTX 3080 / sm_86 class GPUs,
but it does not support newer RTX 50-series GPUs such as RTX 5070 Ti / sm_120.

이 환경은 기존 robodiff 호환성을 위해 PyTorch 1.12.1 + CUDA 11.6을 사용합니다.
RTX 3080 / sm_86 계열에서는 학습 가능하지만, RTX 5070 Ti / sm_120 같은
신형 GPU에서는 CUDA kernel을 실행할 수 없습니다. 그런 경우 remote 3080 장비에서
학습하거나, 별도의 최신 PyTorch 환경을 만들어야 합니다.
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

CPU-only smoke test for checking the code path on unsupported local GPUs:

```bash
python train_can_image.py \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --output data/outputs/can_image_light_debug_cpu \
  --num-epochs 1 \
  --max-train-steps 1 \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu
```

## Train Image World Model

The image world model predicts a multi-frame future visual latent sequence from current multi-view images and a candidate action sequence. It is image-based, but predicts ResNet latents instead of pixels to keep the code path light and stable.

이미지 월드모델은 현재 multi-view 이미지와 candidate action sequence를 입력으로 받아 미래 multi-frame visual latent를 예측합니다. pixel 생성 대신 ResNet latent를 예측해서 구조를 가볍게 유지합니다.

```bash
python train_image_world_model.py \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --output data/outputs/can_image_world_model \
  --device cuda:0 \
  --future-horizon 4 \
  --future-stride 2
```

Smoke test:

```bash
python train_image_world_model.py \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --output data/outputs/can_image_world_model_debug \
  --num-epochs 1 \
  --max-train-steps 1 \
  --batch-size 2 \
  --num-workers 0 \
  --device cuda:0
```

## Train With World Model Conditioning

```bash
python train_can_image.py \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --output data/outputs/can_image_light_wm \
  --device cuda:0 \
  --conditioning obs_wm \
  --world-model-checkpoint data/outputs/can_image_world_model/best.pt \
  --wm-action-mode clean
```

`--wm-action-mode clean` conditions on the expert action sequence during policy training. This is the default DP+WM path. `--wm-action-mode noisy` is only an optional train/eval mismatch ablation; it is not WM-only.

## Evaluate

```bash
python eval_can_image.py \
  --checkpoint data/outputs/can_image_light/latest.pt \
  --output data/eval_can_image_light \
  --device cuda:0 \
  --num-episodes 4
```

Videos are saved for all episodes by default. Use `--save-videos 0` to disable video saving, or `--save-videos N` to save only the first `N` episodes.

Evaluate a world-model-conditioned policy:

```bash
python eval_can_image.py \
  --checkpoint data/outputs/can_image_light_wm/latest.pt \
  --world-model-checkpoint data/outputs/can_image_world_model/best.pt \
  --output data/eval_can_image_light_wm \
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
- World model conditioning is optional and keeps the baseline `obs_only` path unchanged.
