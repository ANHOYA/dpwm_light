from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        scale, bias = embed.chunk(2, dim=1)
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
            self,
            input_dim,
            global_cond_dim,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        cond_dim = dsed + global_cond_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond):
        # sample: (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        global_feature = torch.cat([
            self.diffusion_step_encoder(timestep),
            global_cond,
        ], dim=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


def replace_submodules(root_module, predicate, func):
    if predicate(root_module):
        return func(root_module)
    bn_list = [k.split('.') for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


def get_resnet(name='resnet18', weights=None):
    model = getattr(torchvision.models, name)(weights=weights)
    model.fc = nn.Identity()
    return model


def replace_bn_with_gn(model):
    return replace_submodules(
        root_module=model,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
    )


class CanImageEncoder(nn.Module):
    def __init__(self, camera_keys=('agentview_image', 'robot0_eye_in_hand_image'), lowdim_dim=9):
        super().__init__()
        self.camera_keys = tuple(camera_keys)
        self.lowdim_dim = lowdim_dim
        self.encoders = nn.ModuleDict({
            key: replace_bn_with_gn(get_resnet('resnet18'))
            for key in self.camera_keys
        })
        self.feature_dim = 512 * len(self.camera_keys) + lowdim_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        batch_time = None
        for key in self.camera_keys:
            image = obs[key]
            batch_time = image.shape[:2]
            flat_image = image.flatten(end_dim=1)
            feat = self.encoders[key](flat_image).reshape(*batch_time, -1)
            features.append(feat)
        features.append(obs['lowdim'])
        return torch.cat(features, dim=-1)


class DualConditionFuser(nn.Module):
    def __init__(self, obs_cond_dim, wm_cond_dim, future_horizon, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or obs_cond_dim
        self.net = nn.Sequential(
            nn.Linear(obs_cond_dim + future_horizon * wm_cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, obs_cond_dim),
        )

    def forward(self, obs_cond, wm_cond):
        wm_flat = wm_cond.flatten(start_dim=1)
        return self.net(torch.cat([obs_cond, wm_flat], dim=-1))


def make_policy(obs_horizon, action_dim, lowdim_dim=9,
                camera_keys=('agentview_image', 'robot0_eye_in_hand_image'),
                conditioning='obs_only', wm_cond_dim=0, future_horizon=0):
    obs_encoder = CanImageEncoder(camera_keys=camera_keys, lowdim_dim=lowdim_dim)
    obs_cond_dim = obs_encoder.feature_dim * obs_horizon
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_cond_dim,
    )
    nets = nn.ModuleDict({
        'obs_encoder': obs_encoder,
        'noise_pred_net': noise_pred_net,
    })
    if conditioning == 'obs_wm':
        nets['cond_fuser'] = DualConditionFuser(obs_cond_dim, wm_cond_dim, future_horizon)
    elif conditioning != 'obs_only':
        raise ValueError(f'Unknown conditioning mode: {conditioning}')
    return nets
