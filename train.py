import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, CanImageDataset, CanSequenceDataset
from can_policy import make_policy
from can_utils import resolve_device, save_payload, update_ema_model
from can_world_model import (
    DreamerBCPolicy,
    DreamerRSSM,
    dreamer_kl_loss,
    lambda_returns,
    load_world_model,
    make_dreamer_dp_policy,
    two_hot_loss,
)


def make_loader(dataset, args, device, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )


def batch_obs(batch, device):
    obs = {'lowdim': batch['lowdim'].to(device)} if 'lowdim' in batch else {}
    for key in CAMERA_KEYS:
        obs[key] = batch[key].to(device)
    return obs


def infer_rssm_state(rssm, obs, obs_horizon, action_dim):
    zero = torch.zeros(obs['lowdim'].shape[0], obs_horizon, action_dim, device=obs['lowdim'].device)
    states, _, _, _ = rssm.observe({key: obs[key] for key in CAMERA_KEYS}, zero)
    return states[:, -1]


def raw_rssm_condition(rssm, obs, actions, conditioning):
    state = infer_rssm_state(rssm, obs, obs['lowdim'].shape[1], actions.shape[-1])
    if conditioning == 'rssm_imagine':
        imagined = rssm.imagine(state, actions)
        return torch.cat([state, imagined.flatten(start_dim=1)], dim=-1)
    return state


def diffusion_loss(nets, scheduler, target, cond):
    noise = torch.randn(target.shape, device=target.device)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                              (target.shape[0],), device=target.device).long()
    noisy = scheduler.add_noise(target, noise, timesteps)
    pred = nets['noise_pred_net'](noisy, timesteps, global_cond=cond)
    return nn.functional.mse_loss(pred, noise)


def train_dp(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = CanImageDataset(args.dataset, args.pred_horizon, args.obs_horizon, args.action_horizon)
    loader = make_loader(dataset, args, device)
    example = dataset[0]
    action_dim = example['action'].shape[-1]
    lowdim_dim = example['lowdim'].shape[-1]
    nets = make_policy(args.obs_horizon, action_dim=action_dim, lowdim_dim=lowdim_dim).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_diffusion_iters, beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    ema_nets = copy.deepcopy(nets).eval()
    optim = torch.optim.AdamW(nets.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_scheduler = get_scheduler('cosine', optim, args.lr_warmup_steps, len(loader) * args.num_epochs)
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'lowdim_dim': lowdim_dim, 'camera_keys': list(CAMERA_KEYS)})
    best_loss = float('inf')
    for epoch in tqdm(range(args.num_epochs), desc='DP epoch'):
        losses = []
        for batch in tqdm(loader, desc='Batch', leave=False):
            obs = batch_obs(batch, device)
            target = batch['action'].to(device)
            obs_cond = nets['obs_encoder'](obs).flatten(start_dim=1)
            loss = diffusion_loss(nets, scheduler, target, obs_cond)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            lr_scheduler.step()
            update_ema_model(ema_nets, nets, args.ema_decay)
            losses.append(loss.item())
            if args.max_train_steps is not None and len(losses) >= args.max_train_steps:
                break
        mean_loss = float(np.mean(losses))
        payload = {'model': nets.state_dict(), 'ema_model': ema_nets.state_dict(), 'stats': dataset.stats,
                   'config': config, 'epoch': epoch, 'loss': mean_loss}
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            best_loss = save_payload(output_dir, payload, best_loss)


def train_dreamer_rssm(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = CanSequenceDataset(args.dataset, sequence_length=args.sequence_length, camera_keys=CAMERA_KEYS)
    loader = make_loader(dataset, args, device)
    action_dim = dataset[0]['action'].shape[-1]
    model = DreamerRSSM(action_dim, CAMERA_KEYS, args.embed_dim, args.deter_dim, args.stoch_size,
                        args.classes, args.hidden_dim, args.unimix, args.twohot_bins).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'camera_keys': list(CAMERA_KEYS), 'state_dim': model.state_dim})
    best_loss = float('inf')
    for epoch in tqdm(range(args.num_epochs), desc='Dreamer RSSM epoch'):
        losses = []
        for batch in tqdm(loader, desc='Batch', leave=False):
            obs = {key: batch[key].to(device) for key in CAMERA_KEYS}
            actions = batch['action'].to(device)
            rewards = batch['reward'].to(device)
            done = batch['done'].to(device)
            states, priors, posts, embeds = model.observe(obs, actions)
            heads = model.heads(states)
            embed_loss = F.mse_loss(heads['embed'], embeds.detach())
            reward_loss = two_hot_loss(heads['reward'], rewards, model.support)
            cont_target = 1.0 - done
            cont_loss = F.binary_cross_entropy_with_logits(heads['continue'], cont_target)
            with torch.no_grad():
                values = model.support.decode(heads['value'])
                targets = lambda_returns(rewards, values, cont_target, args.gamma, args.lam)
            value_loss = two_hot_loss(heads['value'], targets, model.support)
            kl, _, _ = dreamer_kl_loss(priors, posts, args.free_nats, args.dyn_scale, args.rep_scale)
            loss = (args.embed_weight * embed_loss + args.reward_weight * reward_loss +
                    args.cont_weight * cont_loss + args.value_weight * value_loss + kl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if args.max_train_steps is not None and len(losses) >= args.max_train_steps:
                break
        mean_loss = float(np.mean(losses))
        payload = {'model': model.state_dict(), 'config': config, 'epoch': epoch, 'loss': mean_loss}
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            best_loss = save_payload(output_dir, payload, best_loss)


def train_rssm_bc(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rssm, rssm_cfg = load_world_model(args.rssm_checkpoint, device)
    for param in rssm.parameters():
        param.requires_grad_(False)
    dataset = CanImageDataset(args.dataset, args.pred_horizon, args.obs_horizon, args.action_horizon)
    loader = make_loader(dataset, args, device)
    action_dim = dataset[0]['action'].shape[-1]
    policy = DreamerBCPolicy(rssm.state_dim, action_dim, args.pred_horizon, args.hidden_dim).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'rssm_config': rssm_cfg, 'state_dim': rssm.state_dim})
    best_loss = float('inf')
    for epoch in tqdm(range(args.num_epochs), desc='RSSM-BC epoch'):
        losses = []
        for batch in tqdm(loader, desc='Batch', leave=False):
            obs = batch_obs(batch, device)
            target = batch['action'].to(device)
            with torch.no_grad():
                state = infer_rssm_state(rssm, obs, args.obs_horizon, action_dim)
            pred = policy(state)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if args.max_train_steps is not None and len(losses) >= args.max_train_steps:
                break
        mean_loss = float(np.mean(losses))
        payload = {'model': policy.state_dict(), 'stats': dataset.stats, 'config': config, 'epoch': epoch, 'loss': mean_loss}
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            best_loss = save_payload(output_dir, payload, best_loss)


def train_dp_rssm(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rssm, rssm_cfg = load_world_model(args.rssm_checkpoint, device)
    for param in rssm.parameters():
        param.requires_grad_(False)
    dataset = CanImageDataset(args.dataset, args.pred_horizon, args.obs_horizon, args.action_horizon)
    loader = make_loader(dataset, args, device)
    action_dim = dataset[0]['action'].shape[-1]
    imagine_horizon = args.pred_horizon if args.conditioning == 'rssm_imagine' else 0
    nets = make_dreamer_dp_policy(rssm.state_dim, action_dim, args.pred_horizon, args.conditioning, imagine_horizon).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_diffusion_iters, beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    ema_nets = copy.deepcopy(nets).eval()
    optim = torch.optim.AdamW(nets.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_scheduler = get_scheduler('cosine', optim, args.lr_warmup_steps, len(loader) * args.num_epochs)
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'rssm_config': rssm_cfg, 'state_dim': rssm.state_dim})
    best_loss = float('inf')
    for epoch in tqdm(range(args.num_epochs), desc='DP-RSSM epoch'):
        losses = []
        for batch in tqdm(loader, desc='Batch', leave=False):
            obs = batch_obs(batch, device)
            target = batch['action'].to(device)
            noise = torch.randn(target.shape, device=device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (target.shape[0],), device=device).long()
            noisy = scheduler.add_noise(target, noise, timesteps)
            cond_actions = target if args.rssm_action_mode == 'clean' else noisy
            with torch.no_grad():
                raw_cond = raw_rssm_condition(rssm, obs, cond_actions, args.conditioning)
            cond = nets['cond_fuser'](raw_cond)
            pred = nets['noise_pred_net'](noisy, timesteps, global_cond=cond)
            loss = nn.functional.mse_loss(pred, noise)
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            lr_scheduler.step()
            update_ema_model(ema_nets, nets, args.ema_decay)
            losses.append(loss.item())
            if args.max_train_steps is not None and len(losses) >= args.max_train_steps:
                break
        mean_loss = float(np.mean(losses))
        payload = {'model': nets.state_dict(), 'ema_model': ema_nets.state_dict(), 'stats': dataset.stats,
                   'config': config, 'epoch': epoch, 'loss': mean_loss}
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            best_loss = save_payload(output_dir, payload, best_loss)


def sample_actions(rssm, nets, obs, cfg, scheduler, sample_iters, device):
    actions = torch.randn((obs['lowdim'].shape[0], cfg['pred_horizon'], cfg['action_dim']), device=device)
    scheduler.set_timesteps(sample_iters)
    for step in scheduler.timesteps:
        with torch.no_grad():
            raw = raw_rssm_condition(rssm, obs, actions, cfg.get('conditioning', 'rssm'))
        cond = nets['cond_fuser'](raw)
        pred = nets['noise_pred_net'](actions, step, global_cond=cond)
        actions = scheduler.step(pred, step, actions).prev_sample
    return actions


def dream_value(rssm, obs, actions):
    with torch.no_grad():
        state = raw_rssm_condition(rssm, obs, actions, 'rssm')
        imagined = rssm.imagine(state, actions)
        heads = rssm.heads(imagined)
        reward = rssm.support.decode(heads['reward'])
        value = rssm.support.decode(heads['value'])
        cont = torch.sigmoid(heads['continue'])
        return (reward + cont * value).mean(dim=1)


def train_dp_rssm_dream(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rssm, rssm_cfg = load_world_model(args.rssm_checkpoint, device)
    for param in rssm.parameters():
        param.requires_grad_(False)
    dataset = CanImageDataset(args.dataset, args.pred_horizon, args.obs_horizon, args.action_horizon)
    loader = make_loader(dataset, args, device)
    action_dim = dataset[0]['action'].shape[-1]
    payload = torch.load(args.policy_checkpoint, map_location=device)
    cfg = payload['config']
    imagine_horizon = cfg['pred_horizon'] if cfg.get('conditioning') == 'rssm_imagine' else 0
    nets = make_dreamer_dp_policy(rssm.state_dim, action_dim, cfg['pred_horizon'], cfg.get('conditioning', 'rssm'), imagine_horizon).to(device)
    nets.load_state_dict(payload.get('ema_model', payload['model']))
    ema_nets = copy.deepcopy(nets).eval()
    optim = torch.optim.AdamW(nets.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = DDPMScheduler(num_train_timesteps=cfg['num_diffusion_iters'], beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'rssm_config': rssm_cfg, 'base_policy_config': cfg,
                   'conditioning': cfg.get('conditioning', 'rssm'), 'pred_horizon': cfg['pred_horizon'],
                   'obs_horizon': cfg['obs_horizon'], 'action_horizon': cfg['action_horizon'],
                   'num_diffusion_iters': cfg['num_diffusion_iters']})
    best_loss = float('inf')
    for epoch in tqdm(range(args.num_epochs), desc='DP-RSSM dream epoch'):
        losses = []
        for batch in tqdm(loader, desc='Batch', leave=False):
            obs = batch_obs(batch, device)
            expert = batch['action'].to(device)
            with torch.no_grad():
                raw = raw_rssm_condition(rssm, obs, expert, cfg.get('conditioning', 'rssm'))
            cond = nets['cond_fuser'](raw)
            bc_loss = diffusion_loss(nets, scheduler, expert, cond)
            with torch.no_grad():
                candidates, scores = [], []
                for _ in range(args.num_candidates):
                    cand = sample_actions(rssm, nets, obs, cfg, scheduler, args.sample_iters, device)
                    candidates.append(cand)
                    scores.append(dream_value(rssm, obs, cand))
                cand_stack = torch.stack(candidates, dim=1)
                score_stack = torch.stack(scores, dim=1)
                best_idx = score_stack.argmax(dim=1)
                best = cand_stack[torch.arange(cand_stack.shape[0], device=device), best_idx]
                raw_best = raw_rssm_condition(rssm, obs, best, cfg.get('conditioning', 'rssm'))
            dream_cond = nets['cond_fuser'](raw_best)
            dream_loss = diffusion_loss(nets, scheduler, best, dream_cond)
            loss = bc_loss + args.dream_loss_weight * dream_loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            update_ema_model(ema_nets, nets, args.ema_decay)
            losses.append(loss.item())
            if args.max_train_steps is not None and len(losses) >= args.max_train_steps:
                break
        mean_loss = float(np.mean(losses))
        payload = {'model': nets.state_dict(), 'ema_model': ema_nets.state_dict(), 'stats': dataset.stats,
                   'config': config, 'epoch': epoch, 'loss': mean_loss}
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            best_loss = save_payload(output_dir, payload, best_loss)


def add_common_args(parser):
    parser.add_argument('--dataset', default='data/robomimic/datasets/can/custom/image.hdf5')
    parser.add_argument('--output', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pred-horizon', type=int, default=16)
    parser.add_argument('--obs-horizon', type=int, default=2)
    parser.add_argument('--action-horizon', type=int, default=8)
    parser.add_argument('--num-diffusion-iters', type=int, default=100)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-warmup-steps', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-train-steps', type=int, default=None)
    parser.add_argument('--ema-decay', type=float, default=0.995)
    parser.add_argument('--rssm-checkpoint', default=None)
    parser.add_argument('--policy-checkpoint', default=None)
    parser.add_argument('--conditioning', choices=['rssm', 'rssm_imagine'], default='rssm')
    parser.add_argument('--rssm-action-mode', choices=['clean', 'noisy'], default='clean')
    parser.add_argument('--sequence-length', type=int, default=32)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--deter-dim', type=int, default=512)
    parser.add_argument('--stoch-size', type=int, default=32)
    parser.add_argument('--classes', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--twohot-bins', type=int, default=255)
    parser.add_argument('--unimix', type=float, default=0.01)
    parser.add_argument('--free-nats', type=float, default=1.0)
    parser.add_argument('--dyn-scale', type=float, default=0.5)
    parser.add_argument('--rep-scale', type=float, default=0.1)
    parser.add_argument('--embed-weight', type=float, default=1.0)
    parser.add_argument('--reward-weight', type=float, default=1.0)
    parser.add_argument('--cont-weight', type=float, default=1.0)
    parser.add_argument('--value-weight', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--grad-clip', type=float, default=100.0)
    parser.add_argument('--num-candidates', type=int, default=4)
    parser.add_argument('--sample-iters', type=int, default=20)
    parser.add_argument('--dream-loss-weight', type=float, default=0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['dp', 'dreamer-rssm', 'rssm-bc', 'dp-rssm', 'dp-rssm-dream'])
    add_common_args(parser)
    args = parser.parse_args()
    if args.output is None:
        args.output = f'data/outputs/{args.method}'
    if args.method in {'rssm-bc', 'dp-rssm', 'dp-rssm-dream'} and args.rssm_checkpoint is None:
        raise ValueError('--rssm-checkpoint is required for RSSM methods')
    if args.method == 'dp-rssm-dream' and args.policy_checkpoint is None:
        raise ValueError('--policy-checkpoint is required for dp-rssm-dream')
    {
        'dp': train_dp,
        'dreamer-rssm': train_dreamer_rssm,
        'rssm-bc': train_rssm_bc,
        'dp-rssm': train_dp_rssm,
        'dp-rssm-dream': train_dp_rssm_dream,
    }[args.method](args)


if __name__ == '__main__':
    main()
