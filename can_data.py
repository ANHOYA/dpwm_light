import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


LOWDIM_KEYS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')
CAMERA_KEYS = ('agentview_image', 'robot0_eye_in_hand_image')
DEFAULT_CAMERA_NAMES = ('agentview', 'robot0_eye_in_hand')


def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        for idx in range(-pad_before, episode_length - sequence_length + pad_after + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            indices.append([buffer_start_idx, buffer_end_idx, start_offset, sequence_length - end_offset])
    return np.asarray(indices, dtype=np.int64)


def sample_sequence(data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = {}
    for key, arr in data.items():
        sample = arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx == 0 and sample_end_idx == sequence_length:
            result[key] = sample
            continue
        out = np.zeros((sequence_length,) + arr.shape[1:], dtype=arr.dtype)
        if sample_start_idx > 0:
            out[:sample_start_idx] = sample[0]
        if sample_end_idx < sequence_length:
            out[sample_end_idx:] = sample[-1]
        out[sample_start_idx:sample_end_idx] = sample
        result[key] = out
    return result


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    return {'min': data.min(axis=0), 'max': data.max(axis=0)}


def normalize_data(data, stats):
    denom = np.maximum(stats['max'] - stats['min'], 1e-6)
    return ((data - stats['min']) / denom) * 2 - 1


def unnormalize_data(ndata, stats):
    denom = np.maximum(stats['max'] - stats['min'], 1e-6)
    return ((ndata + 1) / 2) * denom + stats['min']


def _sorted_demo_keys(data_group):
    return sorted(data_group.keys(), key=lambda x: int(x.split('_')[-1]))


def load_can_hdf5(dataset_path, camera_keys=CAMERA_KEYS, lowdim_keys=LOWDIM_KEYS):
    dataset_path = Path(dataset_path)
    episodes = []
    episode_ends = []
    total = 0
    with h5py.File(dataset_path, 'r') as f:
        for demo_key in _sorted_demo_keys(f['data']):
            demo = f[f'data/{demo_key}']
            obs = demo['obs']
            lowdim = np.concatenate([obs[k][()].astype(np.float32) for k in lowdim_keys], axis=-1)
            ep = {
                'action': demo['actions'][()].astype(np.float32),
                'lowdim': lowdim,
            }
            for key in camera_keys:
                ep[key] = obs[key][()]
            episodes.append(ep)
            total += ep['action'].shape[0]
            episode_ends.append(total)

    data = {}
    for key in ['action', 'lowdim'] + list(camera_keys):
        data[key] = np.concatenate([ep[key] for ep in episodes], axis=0)
    return data, np.asarray(episode_ends, dtype=np.int64)


class CanImageDataset:
    def __init__(self, dataset_path, pred_horizon=16, obs_horizon=2, action_horizon=8,
                 camera_keys=CAMERA_KEYS):
        self.camera_keys = tuple(camera_keys)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        raw_data, episode_ends = load_can_hdf5(dataset_path, camera_keys=self.camera_keys)
        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        self.stats = {
            'action': get_data_stats(raw_data['action']),
            'lowdim': get_data_stats(raw_data['lowdim']),
        }
        self.data = dict(raw_data)
        self.data['action'] = normalize_data(raw_data['action'], self.stats['action']).astype(np.float32)
        self.data['lowdim'] = normalize_data(raw_data['lowdim'], self.stats['lowdim']).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import torch

        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        nsample = sample_sequence(
            self.data,
            self.pred_horizon,
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        )
        out = {
            'action': torch.from_numpy(nsample['action'].astype(np.float32)),
            'lowdim': torch.from_numpy(nsample['lowdim'][:self.obs_horizon].astype(np.float32)),
        }
        for key in self.camera_keys:
            image = nsample[key][:self.obs_horizon]
            image = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0
            out[key] = torch.from_numpy(image)
        return out


def load_raw_metadata(raw_demo):
    with h5py.File(raw_demo, 'r') as f:
        env_meta = json.loads(f['data'].attrs['env_args'])
    return env_meta['env_name'], env_meta['env_kwargs']


def prepare_image_dataset(raw_demo, output_path, camera_names=DEFAULT_CAMERA_NAMES, camera_height=84, camera_width=84):
    import robosuite as suite
    from robomimic.envs.env_base import EnvType
    from robosuite.utils.mjcf_utils import postprocess_model_xml

    raw_demo = Path(raw_demo)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env_name, env_kwargs = load_raw_metadata(raw_demo)
    kwargs = dict(env_kwargs)
    kwargs.update({
        'has_renderer': False,
        'has_offscreen_renderer': True,
        'use_camera_obs': True,
        'camera_names': list(camera_names),
        'camera_heights': camera_height,
        'camera_widths': camera_width,
        'reward_shaping': False,
    })
    env = suite.make(env_name=env_name, **kwargs)
    env_meta = {'env_name': env_name, 'type': EnvType.ROBOSUITE_TYPE, 'env_kwargs': kwargs}

    def set_state(state):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        return env._get_observations(force_update=True)

    def extract(raw_obs):
        obs = {
            'object': np.asarray(raw_obs['object-state']),
            'robot0_eef_pos': np.asarray(raw_obs['robot0_eef_pos']),
            'robot0_eef_quat': np.asarray(raw_obs['robot0_eef_quat']),
            'robot0_gripper_qpos': np.asarray(raw_obs['robot0_gripper_qpos']),
        }
        for cam in camera_names:
            obs[f'{cam}_image'] = np.asarray(raw_obs[f'{cam}_image'][::-1])
        return obs

    with h5py.File(raw_demo, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        demos = _sorted_demo_keys(f_in['data'])
        data_group = f_out.create_group('data')
        total = 0
        for demo_key in tqdm(demos, desc=f'Writing {output_path.name}'):
            demo = f_in[f'data/{demo_key}']
            states = demo['states'][()]
            actions = demo['actions'][()].astype(np.float32)
            model_xml = demo.attrs['model_file'] if 'model_file' in demo.attrs else None
            env.reset()
            if model_xml is not None:
                env.reset_from_xml_string(postprocess_model_xml(model_xml))
                env.sim.reset()
            current_obs = extract(set_state(states[0]))
            traj_obs = {k: [] for k in current_obs}
            traj_next_obs = {k: [] for k in current_obs}
            rewards, dones = [], []
            for t in range(len(actions)):
                if t < len(actions) - 1:
                    next_obs = extract(set_state(states[t + 1]))
                else:
                    set_state(states[t])
                    next_obs = extract(env.step(actions[t])[0])
                for key in traj_obs:
                    traj_obs[key].append(current_obs[key])
                    traj_next_obs[key].append(next_obs[key])
                rewards.append(float(env.reward()))
                dones.append(int(t == len(actions) - 1 or bool(env._check_success())))
                current_obs = next_obs

            ep = data_group.create_group(demo_key)
            ep.create_dataset('actions', data=actions)
            ep.create_dataset('states', data=states)
            ep.create_dataset('rewards', data=np.asarray(rewards, dtype=np.float32))
            ep.create_dataset('dones', data=np.asarray(dones, dtype=np.int32))
            for key, values in traj_obs.items():
                ep.create_dataset(f'obs/{key}', data=np.asarray(values))
                ep.create_dataset(f'next_obs/{key}', data=np.asarray(traj_next_obs[key]))
            if model_xml is not None:
                ep.attrs['model_file'] = model_xml
            ep.attrs['num_samples'] = len(actions)
            total += len(actions)
        data_group.attrs['total'] = total
        data_group.attrs['env_args'] = json.dumps(env_meta, indent=4, default=str)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare-image')
    p.add_argument('--raw-demo', default='data/robomimic/datasets/can/custom/demo.hdf5')
    p.add_argument('--output', default='data/robomimic/datasets/can/custom/image.hdf5')
    p.add_argument('--camera-names', nargs='+', default=list(DEFAULT_CAMERA_NAMES))
    p.add_argument('--camera-height', type=int, default=84)
    p.add_argument('--camera-width', type=int, default=84)
    args = parser.parse_args()
    if args.cmd == 'prepare-image':
        prepare_image_dataset(args.raw_demo, args.output, args.camera_names, args.camera_height, args.camera_width)


if __name__ == '__main__':
    main()
