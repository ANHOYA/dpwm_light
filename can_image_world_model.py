from typing import Dict

import torch
import torch.nn as nn

from can_data import CAMERA_KEYS
from can_policy import get_resnet, replace_bn_with_gn


class MultiViewImageEncoder(nn.Module):
    def __init__(self, camera_keys=CAMERA_KEYS):
        super().__init__()
        self.camera_keys = tuple(camera_keys)
        self.encoders = nn.ModuleDict({
            key: replace_bn_with_gn(get_resnet('resnet18'))
            for key in self.camera_keys
        })
        self.feature_dim = 512 * len(self.camera_keys)

    def forward(self, obs: Dict[str, torch.Tensor], prefix=''):
        features = []
        batch_time = None
        for key in self.camera_keys:
            image = obs[f'{prefix}{key}']
            batch_time = image.shape[:2]
            flat_image = image.flatten(end_dim=1)
            feat = self.encoders[key](flat_image).reshape(*batch_time, -1)
            features.append(feat)
        return torch.cat(features, dim=-1)


class CanImageWorldModel(nn.Module):
    def __init__(self, obs_horizon, pred_horizon, action_dim,
                 future_horizon=4, camera_keys=CAMERA_KEYS, hidden_dim=1024):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.future_horizon = future_horizon
        self.camera_keys = tuple(camera_keys)
        self.image_encoder = MultiViewImageEncoder(camera_keys=self.camera_keys)
        self.latent_dim = self.image_encoder.feature_dim
        input_dim = obs_horizon * self.latent_dim + pred_horizon * action_dim
        output_dim = future_horizon * self.latent_dim
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_observation(self, obs):
        return self.image_encoder(obs)

    def encode_future(self, obs):
        return self.image_encoder(obs, prefix='future_')

    def forward(self, obs, actions):
        obs_latent = self.encode_observation(obs).flatten(start_dim=1)
        action_latent = actions.flatten(start_dim=1)
        pred = self.predictor(torch.cat([obs_latent, action_latent], dim=-1))
        return pred.reshape(actions.shape[0], self.future_horizon, self.latent_dim)


def make_world_model(obs_horizon, pred_horizon, action_dim, future_horizon=4,
                     camera_keys=CAMERA_KEYS, hidden_dim=1024):
    return CanImageWorldModel(
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        future_horizon=future_horizon,
        camera_keys=camera_keys,
        hidden_dim=hidden_dim,
    )


def load_world_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload['config']
    model = make_world_model(
        obs_horizon=cfg['obs_horizon'],
        pred_horizon=cfg['pred_horizon'],
        action_dim=cfg['action_dim'],
        future_horizon=cfg['future_horizon'],
        camera_keys=cfg.get('camera_keys', CAMERA_KEYS),
        hidden_dim=cfg.get('hidden_dim', 1024),
    ).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, cfg
