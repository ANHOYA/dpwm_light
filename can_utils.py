import json
from pathlib import Path

import imageio
import numpy as np
import torch

from can_data import CAMERA_KEYS, DEFAULT_CAMERA_NAMES, LOWDIM_KEYS, normalize_data


def resolve_device(device_arg):
    device = torch.device(device_arg)
    if device.type == 'cpu':
        return device
    if not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU.')
        return torch.device('cpu')
    device_index = 0 if device.index is None else device.index
    major, minor = torch.cuda.get_device_capability(device_index)
    torch_major = int(torch.__version__.split('.')[0])
    if torch_major < 2 and major >= 9:
        raise RuntimeError(
            f'This GPU has compute capability sm_{major}{minor}, but torch {torch.__version__} '
            'does not support it. Use the remote RTX 3080 / sm_86 machine for this old stack, '
            'or create a separate modern PyTorch environment for local training.'
        )
    return device


def update_ema_model(ema_model, model, decay=0.995):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def save_payload(output_dir, payload, best_loss, save_best=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / 'latest.pt')
    if save_best and payload['loss'] < best_loss:
        torch.save(payload, output_dir / 'best.pt')
        return payload['loss']
    return best_loss


def make_eval_env(camera_height=84, camera_width=84):
    import robosuite as suite

    controller_config = suite.load_controller_config(default_controller='OSC_POSE')
    controller_config['kp'] = 300.0
    controller_config['damping_ratio'] = 2.5
    return suite.make(
        env_name='PickPlaceCan',
        robots='Panda',
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(DEFAULT_CAMERA_NAMES),
        camera_heights=camera_height,
        camera_widths=camera_width,
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
    )


def obs_to_model_input(obs, stats):
    lowdim = np.concatenate([obs[k] for k in LOWDIM_KEYS], axis=-1).astype(np.float32)
    result = {'lowdim': normalize_data(lowdim, stats['lowdim']).astype(np.float32)}
    for key in CAMERA_KEYS:
        image = obs[key][::-1]
        result[key] = np.moveaxis(image, -1, 0).astype(np.float32) / 255.0
    return result


def stack_obs(obs_deque, device):
    batch = {'lowdim': []}
    for key in CAMERA_KEYS:
        batch[key] = []
    for obs in obs_deque:
        batch['lowdim'].append(obs['lowdim'])
        for key in CAMERA_KEYS:
            batch[key].append(obs[key])
    out = {'lowdim': torch.from_numpy(np.stack(batch['lowdim'])[None]).to(device)}
    for key in CAMERA_KEYS:
        out[key] = torch.from_numpy(np.stack(batch[key])[None]).to(device)
    return out


def write_eval_log(output_dir, episodes, videos, num_episodes):
    output_dir = Path(output_dir)
    log = {
        'num_episodes': num_episodes,
        'success_rate': float(np.mean([e['success'] for e in episodes])),
        'mean_score': float(np.mean([e['score'] for e in episodes])),
        'mean_steps': float(np.mean([e['steps'] for e in episodes])),
        'mean_action_delta': float(np.mean([e['mean_action_delta'] for e in episodes])),
        'episodes': episodes,
        'videos': videos,
    }
    (output_dir / 'eval_log.json').write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


def maybe_save_video(output_dir, ep, frames, save_videos, fps=10):
    if save_videos >= 0 and ep >= save_videos:
        return ''
    video_path = Path(output_dir) / f'episode_{ep}.mp4'
    imageio.mimsave(video_path, frames, fps=fps)
    return str(video_path)
