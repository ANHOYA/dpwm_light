import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, unnormalize_data
from can_policy import make_policy
from can_utils import make_eval_env, maybe_save_video, obs_to_model_input, stack_obs, write_eval_log
from can_world_model import DreamerBCPolicy, load_world_model, make_dreamer_dp_policy
from train import infer_rssm_state, raw_rssm_condition


def load_dp(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device)
    cfg = payload['config']
    nets = make_policy(cfg['obs_horizon'], cfg['action_dim'], cfg['lowdim_dim']).to(device)
    nets.load_state_dict(payload.get('ema_model', payload['model']))
    nets.eval()
    return nets, cfg, payload['stats']


def load_rssm_bc(checkpoint, rssm, device):
    payload = torch.load(checkpoint, map_location=device)
    cfg = payload['config']
    policy = DreamerBCPolicy(rssm.state_dim, cfg['action_dim'], cfg['pred_horizon'], cfg.get('hidden_dim', 512)).to(device)
    policy.load_state_dict(payload['model'])
    policy.eval()
    return policy, cfg, payload['stats']


def load_dp_rssm(checkpoint, rssm, device):
    payload = torch.load(checkpoint, map_location=device)
    cfg = payload['config']
    imagine_horizon = cfg['pred_horizon'] if cfg.get('conditioning') == 'rssm_imagine' else 0
    nets = make_dreamer_dp_policy(rssm.state_dim, cfg['action_dim'], cfg['pred_horizon'],
                                  cfg.get('conditioning', 'rssm'), imagine_horizon).to(device)
    nets.load_state_dict(payload.get('ema_model', payload['model']))
    nets.eval()
    return nets, cfg, payload['stats']


def sample_dp_actions(nets, cfg, obs, scheduler, device):
    with torch.no_grad():
        obs_cond = nets['obs_encoder'](obs).flatten(start_dim=1)
        actions = torch.randn((1, cfg['pred_horizon'], cfg['action_dim']), device=device)
        scheduler.set_timesteps(cfg['num_diffusion_iters'])
        for step in scheduler.timesteps:
            pred = nets['noise_pred_net'](actions, step, global_cond=obs_cond)
            actions = scheduler.step(pred, step, actions).prev_sample
    return actions


def sample_rssm_actions(rssm, nets, cfg, obs, scheduler, device):
    with torch.no_grad():
        actions = torch.randn((1, cfg['pred_horizon'], cfg['action_dim']), device=device)
        scheduler.set_timesteps(cfg['num_diffusion_iters'])
        for step in scheduler.timesteps:
            raw = raw_rssm_condition(rssm, obs, actions, cfg.get('conditioning', 'rssm'))
            cond = nets['cond_fuser'](raw)
            pred = nets['noise_pred_net'](actions, step, global_cond=cond)
            actions = scheduler.step(pred, step, actions).prev_sample
    return actions


def eval_policy(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    rssm = None
    if args.method in {'rssm-bc', 'dp-rssm', 'dp-rssm-dream'}:
        if args.rssm_checkpoint is None:
            raise ValueError('--rssm-checkpoint is required for RSSM methods')
        rssm, _ = load_world_model(args.rssm_checkpoint, device)

    if args.method == 'dp':
        policy, cfg, stats = load_dp(args.checkpoint, device)
        scheduler = DDPMScheduler(num_train_timesteps=cfg['num_diffusion_iters'], beta_schedule='squaredcos_cap_v2',
                                  clip_sample=True, prediction_type='epsilon')
    elif args.method == 'rssm-bc':
        policy, cfg, stats = load_rssm_bc(args.checkpoint, rssm, device)
        scheduler = None
    else:
        policy, cfg, stats = load_dp_rssm(args.checkpoint, rssm, device)
        scheduler = DDPMScheduler(num_train_timesteps=cfg['num_diffusion_iters'], beta_schedule='squaredcos_cap_v2',
                                  clip_sample=True, prediction_type='epsilon')

    env = make_eval_env(args.camera_height, args.camera_width)
    episodes, videos = [], []
    for ep in range(args.num_episodes):
        obs = env.reset()
        obs_deque = deque([obs_to_model_input(obs, stats)] * cfg['obs_horizon'], maxlen=cfg['obs_horizon'])
        frames = [obs['agentview_image'][::-1]]
        rewards, actions_taken = [], []
        success_step = None
        step_idx = 0
        with tqdm(total=args.max_steps, desc=f'Eval {args.method} episode {ep}', leave=False) as pbar:
            while step_idx < args.max_steps:
                batch_obs = stack_obs(obs_deque, device)
                if args.method == 'dp':
                    naction = sample_dp_actions(policy, cfg, batch_obs, scheduler, device)
                elif args.method == 'rssm-bc':
                    with torch.no_grad():
                        state = infer_rssm_state(rssm, batch_obs, cfg['obs_horizon'], cfg['action_dim'])
                        naction = policy(state)
                else:
                    naction = sample_rssm_actions(rssm, policy, cfg, batch_obs, scheduler, device)
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
        video_path = maybe_save_video(output_dir, ep, frames, args.save_videos)
        if video_path:
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
    write_eval_log(output_dir, episodes, videos, args.num_episodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['dp', 'rssm-bc', 'dp-rssm', 'dp-rssm-dream'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--rssm-checkpoint', default=None)
    parser.add_argument('--output', default='data/eval')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-episodes', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--save-videos', type=int, default=-1)
    parser.add_argument('--camera-height', type=int, default=84)
    parser.add_argument('--camera-width', type=int, default=84)
    eval_policy(parser.parse_args())


if __name__ == '__main__':
    main()
