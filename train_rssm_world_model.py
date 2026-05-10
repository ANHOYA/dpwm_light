import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, CanSequenceDataset
from can_rssm import RSSMWorldModel, kl_loss
from train_can_image import resolve_device


def batch_obs(batch, device):
    return {key: batch[key].to(device) for key in CAMERA_KEYS}


def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = CanSequenceDataset(args.dataset, sequence_length=args.sequence_length, camera_keys=CAMERA_KEYS)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )
    action_dim = dataset[0]['action'].shape[-1]
    model = RSSMWorldModel(
        action_dim=action_dim,
        camera_keys=CAMERA_KEYS,
        embed_dim=args.embed_dim,
        deter_dim=args.deter_dim,
        stoch_dim=args.stoch_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'camera_keys': list(CAMERA_KEYS), 'state_dim': model.state_dim})
    best_loss = float('inf')

    with tqdm(range(args.num_epochs), desc='RSSM epoch') as tglobal:
        for epoch in tglobal:
            losses = []
            for batch in tqdm(loader, desc='Batch', leave=False):
                obs = batch_obs(batch, device)
                actions = batch['action'].to(device)
                done = batch['done'].to(device)
                states, priors, posts, embeds = model.observe(obs, actions)
                pred_embed, pred_cont, pred_success = model.heads(states)
                embed_loss = F.mse_loss(pred_embed, embeds.detach())
                kl = kl_loss(priors, posts)
                cont_target = 1.0 - done
                cont_loss = F.binary_cross_entropy_with_logits(pred_cont, cont_target)
                success_target = done.max(dim=1, keepdim=True).values.expand_as(pred_success)
                success_loss = F.binary_cross_entropy_with_logits(pred_success, success_target)
                loss = embed_loss + args.kl_weight * kl + args.cont_weight * cont_loss + args.success_weight * success_loss
                loss.backward()
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
    parser.add_argument('--output', default='data/outputs/can_rssm_world_model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sequence-length', type=int, default=32)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--deter-dim', type=int, default=512)
    parser.add_argument('--stoch-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--kl-weight', type=float, default=1e-3)
    parser.add_argument('--cont-weight', type=float, default=1.0)
    parser.add_argument('--success-weight', type=float, default=1.0)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-train-steps', type=int, default=None)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
