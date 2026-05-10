from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from can_data import CAMERA_KEYS
from can_policy import ConditionalUnet1D, get_resnet, replace_bn_with_gn


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class TwoHotSupport(nn.Module):
    def __init__(self, bins=255, low=-20.0, high=20.0):
        super().__init__()
        self.bins = bins
        self.low = low
        self.high = high
        self.register_buffer('support', torch.linspace(low, high, bins))

    def encode(self, value):
        value = symlog(value).clamp(self.low, self.high)
        pos = (value - self.low) / (self.high - self.low) * (self.bins - 1)
        lower = pos.floor().long().clamp(0, self.bins - 1)
        upper = (lower + 1).clamp(0, self.bins - 1)
        upper_w = pos - lower.float()
        lower_w = 1.0 - upper_w
        target = torch.zeros(value.shape + (self.bins,), device=value.device)
        target.scatter_add_(-1, lower.unsqueeze(-1), lower_w.unsqueeze(-1))
        target.scatter_add_(-1, upper.unsqueeze(-1), upper_w.unsqueeze(-1))
        return target

    def decode(self, logits):
        probs = logits.softmax(dim=-1)
        return symexp((probs * self.support).sum(dim=-1))


def two_hot_loss(logits, target, support):
    target_dist = support.encode(target)
    return -(target_dist * logits.log_softmax(dim=-1)).sum(dim=-1).mean()


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
            nn.LayerNorm(embed_dim),
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
        return self.net(torch.cat(features, dim=-1))


class DreamerRSSM(nn.Module):
    def __init__(self, action_dim, camera_keys=CAMERA_KEYS, embed_dim=512,
                 deter_dim=512, stoch_size=32, classes=32, hidden_dim=512,
                 unimix=0.01, twohot_bins=255):
        super().__init__()
        self.action_dim = action_dim
        self.camera_keys = tuple(camera_keys)
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_size = stoch_size
        self.classes = classes
        self.stoch_dim = stoch_size * classes
        self.state_dim = deter_dim + self.stoch_dim
        self.unimix = unimix
        self.encoder = MultiViewCNNEncoder(camera_keys=self.camera_keys, embed_dim=embed_dim)
        self.action_proj = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish())
        self.gru = nn.GRUCell(self.stoch_dim + hidden_dim, deter_dim)
        self.prior = nn.Sequential(nn.Linear(deter_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                                   nn.Linear(hidden_dim, stoch_size * classes))
        self.posterior = nn.Sequential(nn.Linear(deter_dim + embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                                       nn.Linear(hidden_dim, stoch_size * classes))
        self.embed_head = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                                        nn.Linear(hidden_dim, embed_dim))
        self.reward_head = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                                         nn.Linear(hidden_dim, twohot_bins))
        self.value_head = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                                        nn.Linear(hidden_dim, twohot_bins))
        self.continue_head = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                                           nn.Linear(hidden_dim, 1))
        self.support = TwoHotSupport(twohot_bins)

    def initial(self, batch_size, device):
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def _logits(self, logits):
        logits = logits.reshape(*logits.shape[:-1], self.stoch_size, self.classes)
        if self.unimix <= 0:
            return logits
        probs = logits.softmax(dim=-1)
        probs = (1.0 - self.unimix) * probs + self.unimix / self.classes
        return torch.log(probs.clamp_min(1e-8))

    def _sample(self, logits):
        logits = self._logits(logits)
        probs = logits.softmax(dim=-1)
        index = torch.distributions.Categorical(probs=probs).sample()
        sample = F.one_hot(index, self.classes).float()
        st_sample = sample + probs - probs.detach()
        return st_sample.flatten(start_dim=-2), logits

    def _state(self, deter, stoch):
        return torch.cat([deter, stoch], dim=-1)

    def obs_step(self, prev, prev_action, embed):
        action = self.action_proj(prev_action)
        deter = self.gru(torch.cat([prev['stoch'], action], dim=-1), prev['deter'])
        prior_logits = self.prior(deter)
        post_logits = self.posterior(torch.cat([deter, embed], dim=-1))
        stoch, post_logits = self._sample(post_logits)
        prior_logits = self._logits(prior_logits)
        return {'deter': deter, 'stoch': stoch}, prior_logits, post_logits

    def img_step(self, prev, prev_action):
        action = self.action_proj(prev_action)
        deter = self.gru(torch.cat([prev['stoch'], action], dim=-1), prev['deter'])
        prior_logits = self.prior(deter)
        stoch, prior_logits = self._sample(prior_logits)
        return {'deter': deter, 'stoch': stoch}, prior_logits

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
        return torch.stack(states, dim=1), torch.stack(priors, dim=1), torch.stack(posts, dim=1), embeds

    def imagine(self, start_state, actions):
        deter, stoch = start_state.split([self.deter_dim, self.stoch_dim], dim=-1)
        prev = {'deter': deter, 'stoch': stoch}
        states = []
        for t in range(actions.shape[1]):
            prev, _ = self.img_step(prev, actions[:, t])
            states.append(self._state(prev['deter'], prev['stoch']))
        return torch.stack(states, dim=1)

    def heads(self, states):
        return {
            'embed': self.embed_head(states),
            'reward': self.reward_head(states),
            'value': self.value_head(states),
            'continue': self.continue_head(states).squeeze(-1),
        }


def categorical_kl(post_logits, prior_logits, free_nats=1.0):
    post_logprob = post_logits.log_softmax(dim=-1)
    prior_logprob = prior_logits.log_softmax(dim=-1)
    post_prob = post_logprob.exp()
    kl = (post_prob * (post_logprob - prior_logprob)).sum(dim=(-1, -2))
    return torch.maximum(kl, torch.ones_like(kl) * free_nats).mean()


def dreamer_kl_loss(prior_logits, post_logits, free_nats=1.0, dyn_scale=0.5, rep_scale=0.1):
    dyn = categorical_kl(post_logits.detach(), prior_logits, free_nats)
    rep = categorical_kl(post_logits, prior_logits.detach(), free_nats)
    return dyn_scale * dyn + rep_scale * rep, dyn.detach(), rep.detach()


def lambda_returns(rewards, values, continues, gamma=0.997, lam=0.95):
    next_values = torch.cat([values[:, 1:], values[:, -1:]], dim=1)
    inputs = rewards + gamma * continues * next_values * (1.0 - lam)
    last = values[:, -1]
    outs = []
    for t in reversed(range(rewards.shape[1])):
        last = inputs[:, t] + gamma * continues[:, t] * lam * last
        outs.append(last)
    return torch.stack(list(reversed(outs)), dim=1)


class DreamerBCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, pred_horizon=16, hidden_dim=512):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
                                 nn.Linear(hidden_dim, pred_horizon * action_dim))

    def forward(self, state):
        return self.net(state).reshape(state.shape[0], self.pred_horizon, self.action_dim)


def make_dreamer_dp_policy(state_dim, action_dim, pred_horizon, conditioning='rssm', imagine_horizon=0):
    cond_dim = state_dim if conditioning == 'rssm' else state_dim * (1 + imagine_horizon)
    fuser = nn.Sequential(nn.Linear(cond_dim, state_dim), nn.LayerNorm(state_dim), nn.Mish(),
                          nn.Linear(state_dim, state_dim))
    return nn.ModuleDict({
        'cond_fuser': fuser,
        'noise_pred_net': ConditionalUnet1D(input_dim=action_dim, global_cond_dim=state_dim),
    })


def load_dreamer_rssm(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload['config']
    model = DreamerRSSM(
        action_dim=cfg['action_dim'],
        camera_keys=cfg.get('camera_keys', CAMERA_KEYS),
        embed_dim=cfg.get('embed_dim', 512),
        deter_dim=cfg.get('deter_dim', 512),
        stoch_size=cfg.get('stoch_size', 32),
        classes=cfg.get('classes', 32),
        hidden_dim=cfg.get('hidden_dim', 512),
        unimix=cfg.get('unimix', 0.01),
        twohot_bins=cfg.get('twohot_bins', 255),
    ).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, cfg
