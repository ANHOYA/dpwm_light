from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from can_data import CAMERA_KEYS
from can_policy import ConditionalUnet1D, get_resnet, replace_bn_with_gn


class MultiViewCNNEncoder(nn.Module):
    def __init__(self, camera_keys=CAMERA_KEYS, embed_dim=512):
        super().__init__()
        self.camera_keys = tuple(camera_keys)
        self.encoders = nn.ModuleDict({
            key: replace_bn_with_gn(get_resnet('resnet18'))
            for key in self.camera_keys
        })
        self.net = nn.Sequential(
            nn.Linear(512 * len(self.camera_keys), embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, obs: Dict[str, torch.Tensor]):
        features = []
        batch_time = None
        for key in self.camera_keys:
            image = obs[key]
            batch_time = image.shape[:2]
            flat = image.flatten(end_dim=1)
            feat = self.encoders[key](flat).reshape(*batch_time, -1)
            features.append(feat)
        fused = torch.cat(features, dim=-1)
        return self.net(fused)


class RSSMWorldModel(nn.Module):
    def __init__(self, action_dim, camera_keys=CAMERA_KEYS, embed_dim=512,
                 deter_dim=512, stoch_dim=64, hidden_dim=512):
        super().__init__()
        self.action_dim = action_dim
        self.camera_keys = tuple(camera_keys)
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.state_dim = deter_dim + stoch_dim
        self.encoder = MultiViewCNNEncoder(camera_keys=self.camera_keys, embed_dim=embed_dim)
        self.action_proj = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.Mish())
        self.gru = nn.GRUCell(stoch_dim + hidden_dim, deter_dim)
        self.prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.embed_head = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )
        self.success_head = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def initial(self, batch_size, device):
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def _dist(self, stats):
        mean, raw_std = stats.chunk(2, dim=-1)
        std = F.softplus(raw_std) + 0.1
        return torch.distributions.Normal(mean, std)

    def _state(self, deter, stoch):
        return torch.cat([deter, stoch], dim=-1)

    def obs_step(self, prev, prev_action, embed):
        action = self.action_proj(prev_action)
        deter = self.gru(torch.cat([prev['stoch'], action], dim=-1), prev['deter'])
        prior = self._dist(self.prior(deter))
        posterior = self._dist(self.posterior(torch.cat([deter, embed], dim=-1)))
        stoch = posterior.rsample()
        return {'deter': deter, 'stoch': stoch}, prior, posterior

    def img_step(self, prev, prev_action):
        action = self.action_proj(prev_action)
        deter = self.gru(torch.cat([prev['stoch'], action], dim=-1), prev['deter'])
        prior = self._dist(self.prior(deter))
        stoch = prior.rsample()
        return {'deter': deter, 'stoch': stoch}, prior

    def observe(self, obs, actions):
        embeds = self.encoder(obs)
        batch_size, horizon = actions.shape[:2]
        prev = self.initial(batch_size, actions.device)
        prev_action = torch.zeros(batch_size, self.action_dim, device=actions.device)
        states, priors, posts = [], [], []
        for t in range(horizon):
            state, prior, post = self.obs_step(prev, prev_action, embeds[:, t])
            states.append(self._state(state['deter'], state['stoch']))
            priors.append(prior)
            posts.append(post)
            prev = state
            prev_action = actions[:, t]
        return torch.stack(states, dim=1), priors, posts, embeds

    def imagine(self, start_state, actions):
        deter, stoch = start_state.split([self.deter_dim, self.stoch_dim], dim=-1)
        prev = {'deter': deter, 'stoch': stoch}
        states = []
        for t in range(actions.shape[1]):
            prev, _ = self.img_step(prev, actions[:, t])
            states.append(self._state(prev['deter'], prev['stoch']))
        return torch.stack(states, dim=1)

    def heads(self, states):
        embed = self.embed_head(states)
        cont = self.continue_head(states).squeeze(-1)
        success = self.success_head(states).squeeze(-1)
        return embed, cont, success


class RSSMBCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, pred_horizon=16, hidden_dim=512):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, pred_horizon * action_dim),
        )

    def forward(self, state):
        return self.net(state).reshape(state.shape[0], self.pred_horizon, self.action_dim)


def make_rssm_policy(state_dim, action_dim, pred_horizon, conditioning='rssm',
                     imagine_horizon=0):
    cond_dim = state_dim
    if conditioning == 'rssm_imagine':
        cond_dim += imagine_horizon * state_dim
    elif conditioning != 'rssm':
        raise ValueError(f'Unknown RSSM policy conditioning: {conditioning}')
    cond_fuser = nn.Sequential(
        nn.Linear(cond_dim, state_dim),
        nn.Mish(),
        nn.Linear(state_dim, state_dim),
    )
    noise_pred_net = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=state_dim)
    return nn.ModuleDict({'cond_fuser': cond_fuser, 'noise_pred_net': noise_pred_net})


def kl_loss(priors, posts):
    losses = [torch.distributions.kl_divergence(post, prior).sum(dim=-1) for prior, post in zip(priors, posts)]
    return torch.stack(losses, dim=1).mean()


def load_rssm(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload['config']
    model = RSSMWorldModel(
        action_dim=cfg['action_dim'],
        camera_keys=cfg.get('camera_keys', CAMERA_KEYS),
        embed_dim=cfg.get('embed_dim', 512),
        deter_dim=cfg.get('deter_dim', 512),
        stoch_dim=cfg.get('stoch_dim', 64),
        hidden_dim=cfg.get('hidden_dim', 512),
    ).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, cfg
