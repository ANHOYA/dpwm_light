# Real Robot Experiment Plan: World-Model-Conditioned Diffusion Policy

Franka + Easy Gripper real robot setup에서 image diffusion policy와 future multi-frame image latent world model을 검증하기 위한 실험 계획서입니다.

This document describes the real robot deployment plan for comparing a baseline image diffusion policy against a world-model-conditioned diffusion policy on a Franka Can manipulation task.

## 1. Goal

목표는 simulation에서 잘 동작한 lightweight image diffusion policy와 image world model dual conditioning 구조를 real Franka 환경으로 옮겨 검증하는 것입니다.

Main comparison:

- Baseline image diffusion policy
- Future-image-latent world-model-conditioned diffusion policy

Key questions:

- 두 카메라 입력만으로 real Can pick-and-place가 안정적으로 가능한가?
- image world model conditioning이 baseline diffusion policy 대비 success rate, time-to-success, action smoothness를 개선하는가?
- sim에서 사용한 dual conditioning 구조가 real data에서도 유지보수 가능한 형태로 동작하는가?

## 2. Real Robot Setup

Hardware:

- Robot: Franka Emika Panda
- Gripper: Easy Gripper
- Wrist camera: gripper 위 또는 gripper 근처에 rigid mount
- Third-person camera: table/workspace를 보는 fixed external camera
- Compute: GPU workstation for training and inference
- Control interface: Franka control stack, ROS bridge, or custom Python control bridge

Recommended camera key mapping:

```text
agentview_image = third-person camera
robot0_eye_in_hand_image = wrist / gripper camera
```

Workspace:

- Tabletop manipulation workspace
- One can object
- Source region for initial can placement
- Target region for final placement
- Fixed or bounded robot home pose

## 3. Task Definition

Task:

```text
Pick up a can from an initial source region and place it into a target region.
```

Episode start:

- Robot starts from a home pose.
- Can is placed in a bounded source region.
- Target region is fixed or visually marked on the table.
- Cameras, lighting, and crop settings remain fixed during each experimental batch.

Success condition:

- Can is inside the target region.
- Can is stably placed or released.
- Robot does not collide with the environment.
- Can is not pushed out of bounds or dropped outside the workspace.

Failure cases:

- Failed grasp
- Can knocked over
- Can pushed out of bounds
- Can dropped before reaching target
- Unsafe robot motion
- Timeout
- Human intervention
- Emergency stop

Initial success labeling can be manual. Automatic success detection can be added later using camera-based object localization or simple workspace markers.

## 4. Observation And Action Space

Observations:

- Third-person RGB image
- Wrist RGB image
- End-effector position
- End-effector orientation
- Gripper state

Image preprocessing:

- Log raw high-resolution frames for analysis.
- Resize/crop policy input to `84x84` to match the current model path.
- Normalize images to `[0, 1]`.
- Keep camera ordering fixed across data collection, training, and deployment.

Action space:

```text
7D action:
dx, dy, dz, droll, dpitch, dyaw, gripper
```

Recommended first real deployment simplification:

- Lock end-effector orientation downward.
- Primarily control delta Cartesian position and gripper.
- Allow yaw only in a later ablation if needed.

Rationale:

- The current teleoperation setup already used downward EEF orientation lock.
- Locked orientation reduces unsafe rotations.
- Translation + gripper is enough for the first Can pick-and-place trials.

## 5. Camera Setup And Calibration

Cameras:

- Wrist camera rigidly mounted near the gripper.
- Third-person camera fixed relative to the table.

Requirements:

- Stable camera pose during collection and evaluation.
- Consistent exposure and white balance.
- Minimal motion blur.
- Fixed crop and resize configuration.
- Camera stream timestamps logged with robot state timestamps.

Calibration plan:

- Intrinsic calibration for both cameras.
- Optional hand-eye calibration for the wrist camera.
- Workspace crop calibration for third-person view.
- First version should not depend on precise 3D reconstruction.

Practical recommendation:

- Use fixed image crops first.
- Keep raw frames for debugging.
- Use the same crop during data collection and deployment.

## 6. Data Collection Plan

Demonstration collection:

- Collect human teleoperation demonstrations using SpaceMouse or another safe teleop interface.
- Record the exact camera streams used by the policy.
- Log robot state, action command, gripper command, timestamps, and images.

Initial target:

- 50 successful demos for pipeline validation.
- 100-200 successful demos for first serious training.
- Add recovery/failure data only after baseline behavior is stable.

Episode metadata:

- Success label
- Initial can pose category
- Target region ID
- Operator notes
- Failure reason if failed
- Camera/crop/calibration version

Demo acceptance criteria:

- Smooth motion
- Successful grasp and placement
- No major camera occlusion
- No emergency stop
- No severe timestamp mismatch
- No accidental collision

## 7. Real Dataset Format

The real dataset should follow the same HDF5 structure as the lightweight simulation dataset whenever possible.

Required fields:

```text
data/demo_*/actions
data/demo_*/obs/agentview_image
data/demo_*/obs/robot0_eye_in_hand_image
data/demo_*/obs/robot0_eef_pos
data/demo_*/obs/robot0_eef_quat
data/demo_*/obs/robot0_gripper_qpos
data/demo_*/next_obs/...
data/demo_*/rewards or success labels
data/demo_*/dones
```

Real-only useful fields:

```text
data/demo_*/timestamps/robot
data/demo_*/timestamps/agentview_camera
data/demo_*/timestamps/wrist_camera
data/demo_*/attrs/calibration_version
data/demo_*/attrs/operator_id
data/demo_*/attrs/notes
data/demo_*/attrs/success
data/demo_*/attrs/failure_reason
```

Reason for matching the sim format:

- Existing `CanImageDataset` can be reused with minimal changes.
- Baseline policy, world model, and eval tooling stay consistent.
- Real and sim datasets can later be mixed or compared cleanly.

## 8. Training And Deployment Stages

Stage 1: Logging and dry-run validation

- Verify camera capture.
- Verify robot state logging.
- Verify action command logging.
- Run policy forward pass without moving the robot.
- Check input shapes and latency.

Stage 2: Baseline real policy

- Train baseline image diffusion policy on real demonstrations.
- Evaluate with conservative action limits.
- Confirm basic pick-and-place capability.

Stage 3: Image world model

- Train future multi-frame visual latent world model on real demonstrations.
- Verify world model train/validation loss.
- Inspect rollouts indirectly through policy performance and failure analysis.

Stage 4: WM-conditioned policy

- Train diffusion policy with frozen image world model conditioning.
- Start with `wm_action_mode=clean`.
- Later compare against `wm_action_mode=noisy`.

Stage 5: Optional sim+real training

- Pretrain on sim.
- Fine-tune on real.
- Compare against real-only training.

## 9. Policy And World Model Architecture

Baseline policy:

```text
current multi-view images + robot lowdim
-> ResNet18-GN image encoder
-> obs_cond
-> ConditionalUnet1D
-> action sequence
```

World model:

```text
current multi-view images + candidate action sequence
-> image world model
-> future multi-frame visual latent sequence
```

WM-conditioned policy:

```text
obs_cond + future visual latent sequence
-> DualConditionFuser
-> ConditionalUnet1D
-> action sequence
```

Recommended initial defaults:

```text
obs_horizon = 2
pred_horizon = 16
future_horizon = 4
future_stride = 2
policy_image_size = 84x84
```

## 10. Control Loop Design

Real deployment loop:

1. Read latest synchronized camera frames.
2. Read latest robot state and gripper state.
3. Crop and resize images to policy input size.
4. Update observation history buffer.
5. Run diffusion policy inference.
6. Clip and validate predicted action sequence.
7. Execute a short action chunk.
8. Replan with fresh observations.

Initial deployment recommendation:

```text
policy frequency = 5-10 Hz
execute 1-2 actions per replan during early testing
increase action chunk length only after stable behavior
```

Important runtime checks:

- Camera frames are fresh.
- Robot state is fresh.
- No NaN or abnormal action values.
- Action stays inside workspace bounds.
- Cartesian velocity and acceleration are below limits.

## 11. Safety Protocol

Safety constraints:

- Workspace bounding box
- Maximum Cartesian velocity
- Maximum Cartesian acceleration
- Maximum rotation command
- Gripper force limit
- Emergency stop accessible
- Human outside robot workspace during autonomous rollout

Runtime stop conditions:

- Action outside workspace bounds
- Stale camera frame
- Stale robot state
- Force/torque threshold exceeded
- Robot enters forbidden zone
- Can leaves workspace
- Operator presses stop
- Policy outputs NaN or extreme values

Deployment sequence:

1. Offline inference on logged data.
2. Live inference without robot motion.
3. Single-step low-speed robot motion.
4. Short-horizon partial rollout.
5. Full rollout at conservative speed.
6. Gradual speed/action-scale increase if safe.

## 12. Evaluation Protocol

Methods to compare:

- Baseline image diffusion policy
- Image world-model-conditioned diffusion policy
- Optional human teleop reference

Evaluation settings:

- Same camera setup
- Same target region
- Same initial object distribution
- Same max episode length
- Same action limits
- Same success criteria

Recommended trial count:

```text
debug: 10-20 trials
first report: 50 trials
final comparison: 100 trials if time allows
```

Metrics:

- Success rate
- Mean score or manual success score
- Time-to-success
- Mean episode steps
- Final reward or final manual score
- Episode return if shaped reward exists
- Action smoothness
- Human intervention count
- Safety stop count
- Failure mode distribution

Suggested summary table:

```text
Method        Success  MeanScore  FinalReward  Return  SuccessStep  ActionDelta  SafetyStops
Baseline DP   0.xx     x.xx       x.xx         x.xx    xx.x         x.xx         x
DP + WM       0.xx     x.xx       x.xx         x.xx    xx.x         x.xx         x
```

## 13. Sim-To-Real Gap And Mitigation

Expected gaps:

- Camera viewpoint mismatch
- Lighting and texture difference
- Object material, mass, and friction mismatch
- Easy Gripper behavior mismatch
- Robot control latency
- Action scale mismatch
- Background clutter
- Real camera exposure and motion blur

Mitigation:

- Match sim and real camera viewpoints where possible.
- Use fixed camera crop and resize.
- Collect real demonstrations with the exact deployment cameras.
- Fine-tune or train on real data before autonomous rollout.
- Use conservative action scaling.
- Add color jitter and crop augmentation during training.
- Log high-resolution videos for failure analysis.

## 14. Experimental Ablations

Main ablations:

- Baseline DP vs DP + image WM
- `wm_action_mode=clean` vs `wm_action_mode=noisy`
- Future horizon `2` vs `4`
- Future stride `1` vs `2`

Camera ablations:

- Third-person only
- Wrist only
- Third-person + wrist

Training ablations:

- Real-only training
- Sim pretraining + real fine-tuning
- Different real demo counts: 50, 100, 200

Control ablations:

- Orientation locked
- Orientation locked with yaw allowed
- Different action chunk execution lengths

## 15. Timeline

Phase 1: Hardware and logging setup

- Mount wrist and third-person cameras.
- Implement synchronized logger.
- Verify robot state/action logging.
- Save raw and policy-resolution images.

Phase 2: Demonstration collection

- Collect 50 successful demos.
- Validate HDF5 dataset format.
- Train baseline policy smoke test.

Phase 3: Baseline real rollout

- Run offline inference.
- Run live inference without motion.
- Run low-speed partial rollouts.
- Run full baseline evaluation.

Phase 4: Image world model

- Train future multi-frame image latent world model.
- Train WM-conditioned policy.
- Run matched evaluation against baseline.

Phase 5: Analysis and iteration

- Analyze videos and failure modes.
- Add demos for weak initial states.
- Tune action scale, crop, and safety filters.
- Repeat final comparison.

## 16. Open Questions

- Which real control interface will be used?
- Will the policy command delta pose, velocity, or target pose?
- What camera model and frame rate will be used?
- How will camera and robot timestamps be synchronized?
- Will success be manually labeled or automatically detected?
- How exactly is the target region defined?
- Should orientation be fully locked or should yaw be allowed?
- Should the first model be trained real-only or sim-pretrained then real-finetuned?
- How many demos are feasible before the first real rollout?

## 17. Immediate Next Steps

1. Mount and verify the two real cameras.
2. Define camera key mapping and image crop.
3. Build real HDF5 logger matching the sim dataset format.
4. Collect 10 dry-run episodes without autonomous policy execution.
5. Collect 50 successful teleop demos.
6. Train baseline image diffusion policy on real demos.
7. Run conservative baseline real rollout.
8. Train image world model and WM-conditioned policy.
9. Compare baseline vs WM with matched initial conditions.
