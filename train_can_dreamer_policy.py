import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, CanImageDataset
from can_dreamer_rssm import load_dreamer_rssm, make_dreamer_dp_policy
from train_can_image import resolve_device, update_ema_model


def infer_state(rssm, obs, obs_horizon, action_dim):
    zero = torch.zeros(obs['lowdim'].shape[0], obs_horizon, action_dim, device=obs['lowdim'].device)
    states, _, _, _ = rssm.observe({key: obs[key] for key in CAMERA_KEYS}, zero)
    return states[:, -1]


def raw_condition(rssm, obs, actions, conditioning):
    state = infer_state(rssm, obs, obs['lowdim'].shape[1], actions.shape[-1])
    if conditioning == 'rssm_imagine':
        imagined = rssm.imagine(state, actions)
        return torch.cat([state, imagined.flatten(start_dim=1)], dim=-1)
    return state


def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rssm, rssm_cfg = load_dreamer_rssm(args.rssm_checkpoint, device)
    for param in rssm.parameters():
        param.requires_grad_(False)

    dataset = CanImageDataset(args.dataset, pred_horizon=args.pred_horizon,
                              obs_horizon=args.obs_horizon, action_horizon=args.action_horizon)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0,
    )
    action_dim = dataset[0]['action'].shape[-1]
    imagine_horizon = args.pred_horizon if args.conditioning == 'rssm_imagine' else 0
    nets = make_dreamer_dp_policy(rssm.state_dim, action_dim, args.pred_horizon,
                                  conditioning=args.conditioning, imagine_horizon=imagine_horizon).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_diffusion_iters, beta_schedule='squaredcos_cap_v2',
                              clip_sample=True, prediction_type='epsilon')
    ema_nets = copy.deepcopy(nets).eval()
    optim = torch.optim.AdamW(nets.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_scheduler = get_scheduler('cosine', optim, args.lr_warmup_steps, len(loader) * args.num_epochs)
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'rssm_config': rssm_cfg, 'state_dim': rssm.state_dim})
    best_loss = float('inf')

    with tqdm(range(args.num_epochs), desc='DP-DreamerRSSM epoch') as tglobal:
        for epoch in tglobal:
            losses = []
            for batch in tqdm(loader, desc='Batch', leave=False):
                obs = {'lowdim': batch['lowdim'].to(device)}
                for key in CAMERA_KEYS:
                    obs[key] = batch[key].to(device)
                target = batch['action'].to(device)
                noise = torch.randn(target.shape, device=device)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                          (target.shape[0],), device=device).long()
                noisy = scheduler.add_noise(target, noise, timesteps)
                cond_actions = target if args.rssm_action_mode == 'clean' else noisy
                with torch.no_grad():
                    raw_cond = raw_condition(rssm, obs, cond_actions, args.conditioning)
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
    parser.add_argument('--output', default='data/outputs/can_dp_dreamer_rssm')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conditioning', choices=['rssm', 'rssm_imagine'], default='rssm')
    parser.add_argument('--rssm-action-mode', choices=['clean', 'noisy'], default='clean')
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
    train(parser.parse_args())


if __name__ == '__main__':
    main()
