# Dream-Conditioned Diffusion Policy: Can Task

Lightweight Can task workspace for comparing original Diffusion Policy and Dreamer-style world-model ablations.

이 repo는 원본 Diffusion Policy의 Hydra/workspace/config 구조를 제거하고, Can task 실험에 필요한 환경, 데이터 수집, 학습, 평가만 유지합니다.

## Environment

This project uses a single conda environment.

```text
Environment name : robodiff-light
Python           : 3.9
PyTorch          : 1.12.1 + CUDA 11.6
Target GPU       : RTX 3080 / sm_86
Local RTX 50xx   : not supported by this old PyTorch CUDA stack
```

Create and activate:

```bash
conda env create -f environment.yaml
conda activate robodiff-light
```

System packages:

```bash
sudo apt update
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

SpaceMouse packages:

```bash
sudo apt install -y libspnav-dev spacenavd
sudo systemctl start spacenavd
```

Verify the environment:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

Important GPU note:

```text
PyTorch 1.12.1 + CUDA 11.6 supports up to sm_86 class GPUs.
Use RTX 3080 for training. RTX 5070 Ti / sm_120 cannot run this stack on CUDA.
```

## Repository Structure

```text
train.py                 # all training modes selected by --method
eval.py                  # all evaluation modes selected by --method
can_data.py              # HDF5 dataset loading and prepare-image command
can_policy.py            # original Diffusion Policy network modules
can_world_model.py       # DreamerV3-style discrete RSSM world model modules
can_utils.py             # device, eval env, logging, video helpers
can_teleop.py            # SpaceMouse teleop and demo collection
environment.yaml         # conda environment definition
```

The old scattered train/eval scripts were intentionally removed. Use `train.py --method ...` and `eval.py --method ...` for every ablation.

## Methods

| Method | World Model | Diffusion | Dream Training | Purpose |
|---|---|---|---|---|
| `dp` | no | yes | no | original Diffusion Policy baseline |
| `dreamer-rssm` | yes | no | no | train DreamerV3-style RSSM world model |
| `rssm-bc` | yes | no | no | RSSM-only behavior cloning baseline |
| `dp-rssm` | yes | yes | no | DP conditioned on RSSM belief / imagination |
| `dp-rssm-dream` | yes | yes | yes | DP-RSSM further trained with dream distillation |

`dp` preserves the original Diffusion Policy structure:

```text
image observation + lowdim -> ResNet encoder -> ConditionalUnet1D action diffusion
```

`dp-rssm` and `dp-rssm-dream` use Dreamer-style RSSM as the world model:

```text
image observation -> RSSM belief state
candidate action sequence -> RSSM imagined future states
belief + imagined future -> diffusion policy condition
```

DreamerV3-style components included in `can_world_model.py`:

```text
discrete categorical stochastic latent
straight-through categorical sampling
unimix categorical smoothing
KL balancing with dynamic and representation terms
free nats
symlog / symexp reward and value support
continuation prediction
RSSM latent imagination
```

## Data Pipeline

Collect raw demonstrations:

```bash
python can_teleop.py --mode collect
```

Default raw output:

```text
data/robomimic/datasets/can/custom/demo.hdf5
```

Convert raw demos to image dataset:

```bash
python can_data.py prepare-image \
  --raw-demo data/robomimic/datasets/can/custom/demo.hdf5 \
  --output data/robomimic/datasets/can/custom/image.hdf5
```

The image dataset contains:

```text
actions
obs/agentview_image
obs/robot0_eye_in_hand_image
obs/robot0_eef_pos
obs/robot0_eef_quat
obs/robot0_gripper_qpos
next_obs/...
rewards
dones
```

## Training

### 1. DP Only

```bash
python train.py \
  --method dp \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --output data/outputs/dp \
  --device cuda:0
```

### 2. Dreamer RSSM World Model

```bash
python train.py \
  --method dreamer-rssm \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --output data/outputs/dreamer_rssm \
  --device cuda:0 \
  --sequence-length 32
```

### 3. RSSM-Only BC

```bash
python train.py \
  --method rssm-bc \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --rssm-checkpoint data/outputs/dreamer_rssm/best.pt \
  --output data/outputs/rssm_bc \
  --device cuda:0
```

### 4. DP + RSSM

```bash
python train.py \
  --method dp-rssm \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --rssm-checkpoint data/outputs/dreamer_rssm/best.pt \
  --output data/outputs/dp_rssm \
  --device cuda:0 \
  --conditioning rssm_imagine
```

Use `--conditioning rssm` for belief-state-only conditioning and `--conditioning rssm_imagine` for belief plus imagined future states.

### 5. DP + RSSM + Dream Training

```bash
python train.py \
  --method dp-rssm-dream \
  --dataset data/robomimic/datasets/can/custom/image.hdf5 \
  --rssm-checkpoint data/outputs/dreamer_rssm/best.pt \
  --policy-checkpoint data/outputs/dp_rssm/best.pt \
  --output data/outputs/dp_rssm_dream \
  --device cuda:0 \
  --num-candidates 4 \
  --dream-loss-weight 0.1
```

## Evaluation

Videos are saved for all episodes by default. Use `--save-videos 0` to disable videos or `--save-videos N` to save only the first `N` episodes.

### DP Only

```bash
python eval.py \
  --method dp \
  --checkpoint data/outputs/dp/latest.pt \
  --output data/eval/dp \
  --device cuda:0 \
  --num-episodes 50
```

### RSSM-Only BC

```bash
python eval.py \
  --method rssm-bc \
  --checkpoint data/outputs/rssm_bc/latest.pt \
  --rssm-checkpoint data/outputs/dreamer_rssm/best.pt \
  --output data/eval/rssm_bc \
  --device cuda:0 \
  --num-episodes 50
```

### DP + RSSM

```bash
python eval.py \
  --method dp-rssm \
  --checkpoint data/outputs/dp_rssm/latest.pt \
  --rssm-checkpoint data/outputs/dreamer_rssm/best.pt \
  --output data/eval/dp_rssm \
  --device cuda:0 \
  --num-episodes 50
```

### DP + RSSM + Dream

```bash
python eval.py \
  --method dp-rssm-dream \
  --checkpoint data/outputs/dp_rssm_dream/latest.pt \
  --rssm-checkpoint data/outputs/dreamer_rssm/best.pt \
  --output data/eval/dp_rssm_dream \
  --device cuda:0 \
  --num-episodes 50
```

Evaluation outputs:

```text
eval_log.json
episode_*.mp4
```

`eval_log.json` includes success rate, mean score, mean steps, action smoothness, per-episode metrics, and saved video paths.

## Recommended Experiment Order

```text
1. train/eval dp
2. train dreamer-rssm
3. train/eval rssm-bc
4. train/eval dp-rssm with --conditioning rssm
5. train/eval dp-rssm with --conditioning rssm_imagine
6. train/eval dp-rssm-dream
```

## CPU Smoke Tests

Use small dimensions for the RSSM smoke test on machines without compatible CUDA.

```bash
python train.py --method dp --device cpu --output /tmp/dp_smoke \
  --num-epochs 1 --max-train-steps 1 --batch-size 1 --num-workers 0
```

```bash
python train.py --method dreamer-rssm --device cpu --output /tmp/rssm_smoke \
  --sequence-length 4 --embed-dim 64 --deter-dim 64 --stoch-size 4 --classes 4 \
  --hidden-dim 64 --twohot-bins 31 --num-epochs 1 --max-train-steps 1 \
  --batch-size 1 --num-workers 0
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
