import argparse
import json
from collections import deque
from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, LOWDIM_KEYS, normalize_data, unnormalize_data
from can_rssm import load_rssm, make_rssm_policy
from eval_can_image import make_eval_env
from train_can_rssm_policy import rssm_condition


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


def policy_cond(rssm, nets, obs, actions, conditioning):
    with torch.no_grad():
        raw_cond = rssm_condition(rssm, obs, actions, conditioning)
    return nets['cond_fuser'](raw_cond)


def eval_policy(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    payload = torch.load(args.checkpoint, map_location=device)
    cfg = payload['config']
    stats = payload['stats']
    rssm, _ = load_rssm(args.rssm_checkpoint or cfg['rssm_checkpoint'], device)
    imagine_horizon = cfg['pred_horizon'] if cfg.get('conditioning') == 'rssm_imagine' else 0
    nets = make_rssm_policy(rssm.state_dim, cfg['action_dim'], cfg['pred_horizon'],
                            conditioning=cfg.get('conditioning', 'rssm'),
                            imagine_horizon=imagine_horizon).to(device)
    nets.load_state_dict(payload.get('ema_model', payload['model']))
    nets.eval()
    scheduler = DDPMScheduler(num_train_timesteps=cfg['num_diffusion_iters'], beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    env = make_eval_env()
    episodes = []
    videos = []
    for ep in range(args.num_episodes):
        obs = env.reset()
        model_obs = obs_to_model_input(obs, stats)
        obs_deque = deque([model_obs] * cfg['obs_horizon'], maxlen=cfg['obs_horizon'])
        frames = [obs['agentview_image'][::-1]]
        rewards, actions_taken = [], []
        success_step = None
        step_idx = 0
        with tqdm(total=args.max_steps, desc=f'RSSM eval episode {ep}', leave=False) as pbar:
            while step_idx < args.max_steps:
                batch_obs = stack_obs(obs_deque, device)
                with torch.no_grad():
                    naction = torch.randn((1, cfg['pred_horizon'], cfg['action_dim']), device=device)
                    scheduler.set_timesteps(cfg['num_diffusion_iters'])
                    for k in scheduler.timesteps:
                        cond = policy_cond(rssm, nets, batch_obs, naction, cfg.get('conditioning', 'rssm'))
                        noise_pred = nets['noise_pred_net'](naction, k, global_cond=cond)
                        naction = scheduler.step(noise_pred, k, naction).prev_sample
                action_pred = unnormalize_data(naction.detach().cpu().numpy()[0], stats['action'])
                start = cfg['obs_horizon'] - 1
                end = start + cfg['action_horizon']
                for action in action_pred[start:end]:
                    obs, reward, _, _ = env.step(action)
                    actions_taken.append(action.astype(np.float32))
                    rewards.append(float(reward))
                    frames.append(obs['agentview_image'][::-1])
                    obs_deque.append(obs_to_model_input(obs, stats))
                    step_idx += 1
                    pbar.update(1)
                    if bool(env._check_success()):
                        success_step = step_idx
                        step_idx = args.max_steps
                        break
                    if step_idx >= args.max_steps:
                        break
        ok = bool(env._check_success())
        action_delta = 0.0
        if len(actions_taken) > 1:
            action_delta = float(np.linalg.norm(np.diff(np.stack(actions_taken), axis=0), axis=-1).mean())
        save_video = args.save_videos < 0 or ep < args.save_videos
        video_path = ''
        if save_video:
            video_path = str(output_dir / f'episode_{ep}.mp4')
            imageio.mimsave(video_path, frames, fps=10)
            videos.append(video_path)
        episodes.append({
            'episode': ep,
            'success': ok,
            'score': max(rewards) if rewards else 0.0,
            'final_reward': rewards[-1] if rewards else 0.0,
            'episode_return': float(np.sum(rewards)),
            'steps': len(rewards),
            'success_step': success_step,
            'mean_action_delta': action_delta,
            'video': video_path,
        })
    env.close()
    log = {
        'num_episodes': args.num_episodes,
        'success_rate': float(np.mean([e['success'] for e in episodes])),
        'mean_score': float(np.mean([e['score'] for e in episodes])),
        'mean_steps': float(np.mean([e['steps'] for e in episodes])),
        'mean_action_delta': float(np.mean([e['mean_action_delta'] for e in episodes])),
        'episodes': episodes,
        'videos': videos,
    }
    (output_dir / 'eval_log.json').write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--rssm-checkpoint', default=None)
    parser.add_argument('--output', default='data/eval_can_dp_rssm')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-episodes', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--save-videos', type=int, default=-1)
    eval_policy(parser.parse_args())


if __name__ == '__main__':
    main()
