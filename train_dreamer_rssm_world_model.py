import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, CanSequenceDataset
from can_dreamer_rssm import DreamerRSSM, dreamer_kl_loss, lambda_returns, two_hot_loss
from train_can_image import resolve_device


def batch_obs(batch, device):
    return {key: batch[key].to(device) for key in CAMERA_KEYS}


def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = CanSequenceDataset(args.dataset, sequence_length=args.sequence_length, camera_keys=CAMERA_KEYS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0,
    )
    action_dim = dataset[0]['action'].shape[-1]
    model = DreamerRSSM(
        action_dim=action_dim,
        camera_keys=CAMERA_KEYS,
        embed_dim=args.embed_dim,
        deter_dim=args.deter_dim,
        stoch_size=args.stoch_size,
        classes=args.classes,
        hidden_dim=args.hidden_dim,
        unimix=args.unimix,
        twohot_bins=args.twohot_bins,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'camera_keys': list(CAMERA_KEYS), 'state_dim': model.state_dim})
    best_loss = float('inf')

    with tqdm(range(args.num_epochs), desc='Dreamer RSSM epoch') as tglobal:
        for epoch in tglobal:
            losses = []
            for batch in tqdm(loader, desc='Batch', leave=False):
                obs = batch_obs(batch, device)
                actions = batch['action'].to(device)
                rewards = batch['reward'].to(device)
                done = batch['done'].to(device)
                states, priors, posts, embeds = model.observe(obs, actions)
                heads = model.heads(states)
                embed_loss = F.mse_loss(heads['embed'], embeds.detach())
                reward_loss = two_hot_loss(heads['reward'], rewards, model.support)
                continue_target = 1.0 - done
                continue_loss = F.binary_cross_entropy_with_logits(heads['continue'], continue_target)
                with torch.no_grad():
                    values = model.support.decode(heads['value'])
                    targets = lambda_returns(rewards, values, continue_target, args.gamma, args.lam)
                value_loss = two_hot_loss(heads['value'], targets, model.support)
                kl, dyn_kl, rep_kl = dreamer_kl_loss(
                    priors, posts,
                    free_nats=args.free_nats,
                    dyn_scale=args.dyn_scale,
                    rep_scale=args.rep_scale,
                )
                loss = (args.embed_weight * embed_loss + args.reward_weight * reward_loss +
                        args.cont_weight * continue_loss + args.value_weight * value_loss + kl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                losses.append(loss.item())
                if args.max_train_steps is not None and len(losses) >= args.max_train_steps:
                    break
            mean_loss = float(np.mean(losses))
            tglobal.set_postfix(loss=mean_loss)
            if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
                payload = {'model': model.state_dict(), 'config': config, 'epoch': epoch, 'loss': mean_loss}
                torch.save(payload, output_dir / 'latest.pt')
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(payload, output_dir / 'best.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/robomimic/datasets/can/custom/image.hdf5')
    parser.add_argument('--output', default='data/outputs/can_dreamer_rssm_world_model')
    parser.add_argument('--device', default='cuda:0')
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
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-train-steps', type=int, default=None)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
