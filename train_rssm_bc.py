import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, CanImageDataset
from can_rssm import RSSMBCPolicy, load_rssm
from train_can_image import resolve_device


def infer_rssm_state(rssm, obs, obs_horizon, action_dim):
    zero_actions = torch.zeros(obs['lowdim'].shape[0], obs_horizon, action_dim, device=obs['lowdim'].device)
    states, _, _, _ = rssm.observe({key: obs[key] for key in CAMERA_KEYS}, zero_actions)
    return states[:, -1]


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
    policy = RSSMBCPolicy(rssm.state_dim, action_dim, args.pred_horizon, args.hidden_dim).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'rssm_config': rssm_cfg, 'state_dim': rssm.state_dim})
    best_loss = float('inf')

    with tqdm(range(args.num_epochs), desc='RSSM-BC epoch') as tglobal:
        for epoch in tglobal:
            losses = []
            for batch in tqdm(loader, desc='Batch', leave=False):
                obs = {'lowdim': batch['lowdim'].to(device)}
                for key in CAMERA_KEYS:
                    obs[key] = batch[key].to(device)
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
            tglobal.set_postfix(loss=mean_loss)
            if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
                payload = {'model': policy.state_dict(), 'config': config, 'epoch': epoch, 'loss': mean_loss}
                torch.save(payload, output_dir / 'latest.pt')
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(payload, output_dir / 'best.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/robomimic/datasets/can/custom/image.hdf5')
    parser.add_argument('--rssm-checkpoint', required=True)
    parser.add_argument('--output', default='data/outputs/can_rssm_bc')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pred-horizon', type=int, default=16)
    parser.add_argument('--obs-horizon', type=int, default=2)
    parser.add_argument('--action-horizon', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-train-steps', type=int, default=None)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
