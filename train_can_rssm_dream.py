import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, CanImageDataset
from can_rssm import load_rssm, make_rssm_policy
from train_can_image import resolve_device, update_ema_model
from train_can_rssm_policy import rssm_condition


def load_policy(path, rssm, action_dim, device):
    payload = torch.load(path, map_location=device)
    cfg = payload['config']
    imagine_horizon = cfg['pred_horizon'] if cfg.get('conditioning') == 'rssm_imagine' else 0
    nets = make_rssm_policy(
        rssm.state_dim,
        action_dim,
        cfg['pred_horizon'],
        conditioning=cfg.get('conditioning', 'rssm'),
        imagine_horizon=imagine_horizon,
    ).to(device)
    nets.load_state_dict(payload.get('ema_model', payload['model']))
    return nets, cfg


def policy_cond(rssm, nets, obs, actions, conditioning):
    with torch.no_grad():
        raw_cond = rssm_condition(rssm, obs, actions, conditioning)
    return nets['cond_fuser'](raw_cond)


def sample_actions(rssm, nets, obs, cfg, scheduler, num_iters, device):
    actions = torch.randn((obs['lowdim'].shape[0], cfg['pred_horizon'], cfg['action_dim']), device=device)
    scheduler.set_timesteps(num_iters)
    for step in scheduler.timesteps:
        cond = policy_cond(rssm, nets, obs, actions, cfg.get('conditioning', 'rssm'))
        noise_pred = nets['noise_pred_net'](actions, step, global_cond=cond)
        actions = scheduler.step(noise_pred, step, actions).prev_sample
    return actions


def dream_scores(rssm, obs, actions):
    with torch.no_grad():
        state = rssm_condition(rssm, obs, actions, 'rssm')
        imagined = rssm.imagine(state, actions)
        _, _, success_logits = rssm.heads(imagined)
        return torch.sigmoid(success_logits[:, -1])


def diffusion_loss(nets, scheduler, target, cond):
    noise = torch.randn(target.shape, device=target.device)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                              (target.shape[0],), device=target.device).long()
    noisy = scheduler.add_noise(target, noise, timesteps)
    pred = nets['noise_pred_net'](noisy, timesteps, global_cond=cond)
    return nn.functional.mse_loss(pred, noise)


def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rssm, rssm_cfg = load_rssm(args.rssm_checkpoint, device)
    for param in rssm.parameters():
        param.requires_grad_(False)

    dataset = CanImageDataset(args.dataset, pred_horizon=args.pred_horizon,
                              obs_horizon=args.obs_horizon, action_horizon=args.action_horizon)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )
    action_dim = dataset[0]['action'].shape[-1]
    nets, cfg = load_policy(args.policy_checkpoint, rssm, action_dim, device)
    ema_nets = copy.deepcopy(nets).eval()
    optim = torch.optim.AdamW(nets.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = DDPMScheduler(num_train_timesteps=cfg['num_diffusion_iters'], beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'rssm_config': rssm_cfg, 'base_policy_config': cfg})
    best_loss = float('inf')

    with tqdm(range(args.num_epochs), desc='DP-RSSM dream epoch') as tglobal:
        for epoch in tglobal:
            losses = []
            for batch in tqdm(loader, desc='Batch', leave=False):
                obs = {'lowdim': batch['lowdim'].to(device)}
                for key in CAMERA_KEYS:
                    obs[key] = batch[key].to(device)
                expert = batch['action'].to(device)
                cond = policy_cond(rssm, nets, obs, expert, cfg.get('conditioning', 'rssm'))
                bc_loss = diffusion_loss(nets, scheduler, expert, cond)

                candidates = []
                scores = []
                with torch.no_grad():
                    for _ in range(args.num_candidates):
                        cand = sample_actions(rssm, nets, obs, cfg, scheduler, args.sample_iters, device)
                        candidates.append(cand)
                        scores.append(dream_scores(rssm, obs, cand))
                    cand_stack = torch.stack(candidates, dim=1)
                    score_stack = torch.stack(scores, dim=1)
                    best_idx = score_stack.argmax(dim=1)
                    best = cand_stack[torch.arange(cand_stack.shape[0], device=device), best_idx]
                dream_cond = policy_cond(rssm, nets, obs, best, cfg.get('conditioning', 'rssm'))
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
            tglobal.set_postfix(loss=mean_loss)
            if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
                payload = {'model': nets.state_dict(), 'ema_model': ema_nets.state_dict(),
                           'stats': dataset.stats, 'config': config, 'epoch': epoch, 'loss': mean_loss}
                torch.save(payload, output_dir / 'latest.pt')
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(payload, output_dir / 'best.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/robomimic/datasets/can/custom/image.hdf5')
    parser.add_argument('--rssm-checkpoint', required=True)
    parser.add_argument('--policy-checkpoint', required=True)
    parser.add_argument('--output', default='data/outputs/can_dp_rssm_dream')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pred-horizon', type=int, default=16)
    parser.add_argument('--obs-horizon', type=int, default=2)
    parser.add_argument('--action-horizon', type=int, default=8)
    parser.add_argument('--num-candidates', type=int, default=4)
    parser.add_argument('--sample-iters', type=int, default=20)
    parser.add_argument('--dream-loss-weight', type=float, default=0.1)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-train-steps', type=int, default=None)
    parser.add_argument('--ema-decay', type=float, default=0.995)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
