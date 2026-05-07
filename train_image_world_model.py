import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from can_data import CAMERA_KEYS, CanImageDataset
from can_image_world_model import make_world_model
from train_can_image import resolve_device


def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = CanImageDataset(
        dataset_path=args.dataset,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        camera_keys=CAMERA_KEYS,
        return_future_images=True,
        future_horizon=args.future_horizon,
        future_stride=args.future_stride,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
    )

    example = dataset[0]
    action_dim = example['action'].shape[-1]
    model = make_world_model(
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_dim=action_dim,
        future_horizon=args.future_horizon,
        camera_keys=CAMERA_KEYS,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    config = vars(args).copy()
    config.update({
        'action_dim': action_dim,
        'camera_keys': list(CAMERA_KEYS),
        'latent_dim': model.latent_dim,
    })
    best_loss = float('inf')

    with tqdm(range(args.num_epochs), desc='World model epoch') as tglobal:
        for epoch in tglobal:
            epoch_loss = []
            for batch in tqdm(dataloader, desc='Batch', leave=False):
                obs = {}
                for key in CAMERA_KEYS:
                    obs[key] = batch[key].to(device)
                    obs[f'future_{key}'] = batch[f'future_{key}'].to(device)
                actions = batch['action'].to(device)

                pred = model(obs, actions)
                with torch.no_grad():
                    target = model.encode_future(obs)
                loss = nn.functional.mse_loss(pred, target)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_loss.append(loss.item())
                if args.max_train_steps is not None and len(epoch_loss) >= args.max_train_steps:
                    break

            mean_loss = float(np.mean(epoch_loss))
            tglobal.set_postfix(loss=mean_loss)
            if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
                payload = {
                    'model': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'loss': mean_loss,
                }
                torch.save(payload, output_dir / 'latest.pt')
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    torch.save(payload, output_dir / 'best.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/robomimic/datasets/can/custom/image.hdf5')
    parser.add_argument('--output', default='data/outputs/can_image_world_model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pred-horizon', type=int, default=16)
    parser.add_argument('--obs-horizon', type=int, default=2)
    parser.add_argument('--action-horizon', type=int, default=8)
    parser.add_argument('--future-horizon', type=int, default=4)
    parser.add_argument('--future-stride', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-train-steps', type=int, default=None)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
