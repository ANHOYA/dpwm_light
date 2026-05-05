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
from can_policy import make_policy


def update_ema_model(ema_model, model, decay=0.995):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def resolve_device(device_arg):
    device = torch.device(device_arg)
    if device.type == 'cpu':
        return device
    if not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU.')
        return torch.device('cpu')
    device_index = 0 if device.index is None else device.index
    major, minor = torch.cuda.get_device_capability(device_index)
    torch_major = int(torch.__version__.split('.')[0])
    if torch_major < 2 and major >= 9:
        raise RuntimeError(
            f'This GPU has compute capability sm_{major}{minor}, but torch {torch.__version__} '
            'does not support it. Use the remote RTX 3080 / sm_86 machine for this '
            'old robodiff stack, or create a separate modern PyTorch environment for local training.'
        )
    return device


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
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    example = dataset[0]
    action_dim = example['action'].shape[-1]
    lowdim_dim = example['lowdim'].shape[-1]
    nets = make_policy(args.obs_horizon, action_dim=action_dim, lowdim_dim=lowdim_dim).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    ema_nets = copy.deepcopy(nets).eval()
    optimizer = torch.optim.AdamW(nets.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=len(dataloader) * args.num_epochs,
    )

    config = vars(args).copy()
    config.update({'action_dim': action_dim, 'lowdim_dim': lowdim_dim, 'camera_keys': list(CAMERA_KEYS)})
    best_loss = float('inf')

    with tqdm(range(args.num_epochs), desc='Epoch') as tglobal:
        for epoch in tglobal:
            epoch_loss = []
            for batch in tqdm(dataloader, desc='Batch', leave=False):
                obs = {'lowdim': batch['lowdim'].to(device)}
                for key in CAMERA_KEYS:
                    obs[key] = batch[key].to(device)
                naction = batch['action'].to(device)
                batch_size = naction.shape[0]

                obs_features = nets['obs_encoder'](obs)
                obs_cond = obs_features.flatten(start_dim=1)

                noise = torch.randn(naction.shape, device=device)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=device,
                ).long()
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                noise_pred = nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
                loss = nn.functional.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                update_ema_model(ema_nets, nets, decay=args.ema_decay)
                epoch_loss.append(loss.item())
                if args.max_train_steps is not None and len(epoch_loss) >= args.max_train_steps:
                    break

            mean_loss = float(np.mean(epoch_loss))
            tglobal.set_postfix(loss=mean_loss)
            if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
                payload = {
                    'model': nets.state_dict(),
                    'ema_model': ema_nets.state_dict(),
                    'stats': dataset.stats,
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
    parser.add_argument('--output', default='data/outputs/can_image_light')
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
    train(parser.parse_args())


if __name__ == '__main__':
    main()
