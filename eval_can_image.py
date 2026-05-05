import argparse
import json
from collections import deque
from pathlib import Path

import imageio
import numpy as np
import robosuite as suite
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, LOWDIM_KEYS, normalize_data, unnormalize_data
from can_policy import make_policy


def make_eval_env(camera_height=84, camera_width=84):
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
        camera_names=list(CAMERA_KEYS),
        camera_heights=camera_height,
        camera_widths=camera_width,
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
    )


def obs_to_model_input(obs, stats):
    lowdim = np.concatenate([obs[k] for k in LOWDIM_KEYS], axis=-1).astype(np.float32)
    nlowdim = normalize_data(lowdim, stats['lowdim']).astype(np.float32)
    result = {'lowdim': nlowdim}
    for key in CAMERA_KEYS:
        image = obs[key][::-1]
        image = np.moveaxis(image, -1, 0).astype(np.float32) / 255.0
        result[key] = image
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


def eval_policy(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    payload = torch.load(args.checkpoint, map_location=device)
    cfg = payload['config']
    stats = payload['stats']
    nets = make_policy(cfg['obs_horizon'], cfg['action_dim'], cfg['lowdim_dim']).to(device)
    nets.load_state_dict(payload.get('ema_model', payload['model']))
    nets.eval()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg['num_diffusion_iters'],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    env = make_eval_env()
    scores = []
    success = []
    all_videos = []
    for ep in range(args.num_episodes):
        obs = env.reset()
        model_obs = obs_to_model_input(obs, stats)
        obs_deque = deque([model_obs] * cfg['obs_horizon'], maxlen=cfg['obs_horizon'])
        rewards = []
        frames = [obs['agentview_image'][::-1]]
        step_idx = 0
        done = False
        with tqdm(total=args.max_steps, desc=f'Eval episode {ep}', leave=False) as pbar:
            while not done and step_idx < args.max_steps:
                batch_obs = stack_obs(obs_deque, device)
                with torch.no_grad():
                    obs_features = nets['obs_encoder'](batch_obs)
                    obs_cond = obs_features.flatten(start_dim=1)
                    naction = torch.randn((1, cfg['pred_horizon'], cfg['action_dim']), device=device)
                    noise_scheduler.set_timesteps(cfg['num_diffusion_iters'])
                    for k in noise_scheduler.timesteps:
                        noise_pred = nets['noise_pred_net'](naction, k, global_cond=obs_cond)
                        naction = noise_scheduler.step(noise_pred, k, naction).prev_sample

                naction = naction.detach().cpu().numpy()[0]
                action_pred = unnormalize_data(naction, stats['action'])
                start = cfg['obs_horizon'] - 1
                end = start + cfg['action_horizon']
                for action in action_pred[start:end]:
                    obs, reward, _, info = env.step(action)
                    model_obs = obs_to_model_input(obs, stats)
                    obs_deque.append(model_obs)
                    rewards.append(float(reward))
                    frames.append(obs['agentview_image'][::-1])
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx >= args.max_steps or bool(env._check_success()):
                        done = True
                        break
        score = max(rewards) if rewards else 0.0
        ok = bool(env._check_success())
        scores.append(score)
        success.append(ok)
        if ep < args.save_videos:
            video_path = output_dir / f'episode_{ep}.mp4'
            imageio.mimsave(video_path, frames, fps=10)
            all_videos.append(str(video_path))

    env.close()
    log = {
        'num_episodes': args.num_episodes,
        'mean_score': float(np.mean(scores)),
        'success_rate': float(np.mean(success)),
        'scores': scores,
        'success': success,
        'videos': all_videos,
    }
    (output_dir / 'eval_log.json').write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='data/outputs/can_image_light/latest.pt')
    parser.add_argument('--output', default='data/eval_can_image_light')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-episodes', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--save-videos', type=int, default=2)
    eval_policy(parser.parse_args())


if __name__ == '__main__':
    main()
