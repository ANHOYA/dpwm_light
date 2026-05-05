import argparse
import json
import os
import shutil
import sys
import time
from threading import Event, Thread

import cv2
import h5py
import hid
import numpy as np
import robosuite as suite
from robomimic.envs.env_base import EnvType
from robosuite.devices import Device
from robosuite.utils.input_utils import input2action
from robosuite.utils.transform_utils import rotation_matrix


DEFAULT_EEF_DOWN_ROTATION = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
])


class LinuxSpaceMouse(Device):
    def __init__(self, pos_sensitivity=0.005, rot_sensitivity=0.001,
                 vendor_id=9583, product_id=50741, lock_orientation=True):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.lock_orientation = lock_orientation
        self.rotation = DEFAULT_EEF_DOWN_ROTATION.copy()
        self._control = np.zeros(6, dtype=np.float64)
        self._left_pressed = False
        self._right_pressed = False
        self._reset_latch = False
        self._enabled = False
        self._stop_event = Event()
        self._device = self._open_device()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _to_int16(b1, b2):
        value = b1 | (b2 << 8)
        return value - 65536 if value >= 32768 else value

    def _open_device(self):
        devices = [
            d for d in hid.enumerate()
            if d['vendor_id'] == self.vendor_id and d['product_id'] == self.product_id
        ]
        if not devices:
            raise RuntimeError('SpaceMouse Compact not found')
        device = hid.device()
        device.open_path(devices[0]['path'])
        return device

    def _run(self):
        while not self._stop_event.is_set():
            data = self._device.read(7, 50)
            if not data:
                continue
            report_id = data[0]
            if report_id == 1:
                self._control[1] = self._to_int16(data[1], data[2]) / 350.0
                self._control[0] = self._to_int16(data[3], data[4]) / 350.0
                self._control[2] = -self._to_int16(data[5], data[6]) / 350.0
            elif report_id == 2:
                self._control[3] = self._to_int16(data[1], data[2]) / 350.0
                self._control[4] = self._to_int16(data[3], data[4]) / 350.0
                self._control[5] = self._to_int16(data[5], data[6]) / 350.0
            elif report_id == 3 and len(data) >= 2:
                buttons = data[1]
                self._left_pressed = bool(buttons & (1 << 0))
                self._right_pressed = bool(buttons & (1 << 1))

    def close(self):
        self._stop_event.set()
        try:
            self._thread.join(timeout=1.0)
            self._device.close()
        except Exception:
            pass

    def start_control(self):
        self.rotation = DEFAULT_EEF_DOWN_ROTATION.copy()
        self._enabled = True
        self._reset_latch = False

    def get_controller_state(self):
        dpos = self._control[:3].copy()
        raw_drotation = self._control[3:].copy()
        if self.lock_orientation:
            raw_drotation[:] = 0.0

        roll, pitch, yaw = raw_drotation * 0.005 * self.rot_sensitivity
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        reset = 0
        if self._right_pressed and not self._reset_latch:
            reset = 1
            self._reset_latch = True
            self._enabled = False
        elif not self._right_pressed:
            self._reset_latch = False

        return {
            'dpos': dpos * self.pos_sensitivity,
            'rotation': self.rotation,
            'raw_drotation': raw_drotation,
            'grasp': 1.0 if self._left_pressed else 0.0,
            'reset': reset,
        }


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def make_env(args):
    controller_config = suite.load_controller_config(default_controller='OSC_POSE')
    controller_config['kp'] = args.controller_kp
    controller_config['damping_ratio'] = args.controller_damping
    env_kwargs = {
        'robots': args.robot,
        'controller_configs': controller_config,
        'control_freq': args.control_freq,
        'reward_shaping': True,
        'ignore_done': True,
    }
    env = suite.make(
        env_name=args.env,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=args.camera,
        camera_widths=args.width,
        camera_heights=args.height,
        **env_kwargs,
    )
    return env, env_kwargs


def archive_existing_hdf5s(output_path):
    parent = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(parent):
        return None
    files = [os.path.join(parent, n) for n in os.listdir(parent) if n.endswith('.hdf5')]
    if not files:
        return None
    archive_dir = os.path.join(parent, 'archive_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(archive_dir, exist_ok=False)
    for file_path in sorted(files):
        shutil.move(file_path, os.path.join(archive_dir, os.path.basename(file_path)))
    return archive_dir


def ensure_demo_file(output_path, env_name, env_kwargs):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    h5_file = h5py.File(output_path, 'a')
    data_group = h5_file.require_group('data')
    env_meta = {'env_name': env_name, 'type': EnvType.ROBOSUITE_TYPE, 'env_kwargs': to_jsonable(env_kwargs)}
    data_group.attrs['env_args'] = json.dumps(env_meta, indent=4)
    data_group.attrs['total'] = int(data_group.attrs.get('total', 0))
    return h5_file, data_group


def next_demo_index(data_group):
    ids = []
    for key in data_group.keys():
        if key.startswith('demo_'):
            ids.append(int(key.split('_')[-1]))
    return 0 if not ids else max(ids) + 1


def save_episode(data_group, demo_index, model_xml, states, actions):
    if len(actions) == 0:
        return False
    ep = data_group.create_group(f'demo_{demo_index}')
    ep.create_dataset('actions', data=np.asarray(actions, dtype=np.float32))
    ep.create_dataset('states', data=np.asarray(states, dtype=np.float64))
    ep.attrs['model_file'] = model_xml
    ep.attrs['num_samples'] = len(actions)
    data_group.attrs['total'] = int(data_group.attrs.get('total', 0)) + len(actions)
    return True


def reset_episode(env, device):
    obs = env.reset()
    device.start_control()
    return obs, env.sim.model.get_xml(), np.asarray(env.sim.get_state().flatten()), [], []


def draw_overlay(frame, lines):
    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24


def run(args):
    if args.mode == 'collect' and args.overwrite:
        archive_dir = archive_existing_hdf5s(args.output)
        if archive_dir:
            print(f'Archived existing hdf5 files to: {archive_dir}')

    env, env_kwargs = make_env(args)
    device = LinuxSpaceMouse(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
        vendor_id=args.vendor_id,
        product_id=args.product_id,
        lock_orientation=not args.free_rotation,
    )
    h5_file = None
    try:
        if args.mode == 'collect':
            h5_file, data_group = ensure_demo_file(args.output, args.env, env_kwargs)
            demo_index = next_demo_index(data_group)
        else:
            data_group = None
            demo_index = 0

        obs, model_xml, current_state, states, actions = reset_episode(env, device)
        print('Mode:', args.mode)
        print('Left button: gripper close | Right button: reset/discard | q: quit')
        print('Orientation lock:', not args.free_rotation)
        dt = 1.0 / args.control_freq
        status = ''

        while True:
            frame = cv2.flip(obs[f'{args.camera}_image'][:, :, ::-1], 0)
            success = bool(env._check_success())
            lines = [
                f'mode: {args.mode}',
                f'steps: {len(actions)} success: {success}',
                'q quit | r reset | right reset | left gripper',
            ]
            if args.mode == 'collect':
                lines.insert(1, f'saved demos: {demo_index}')
                lines.append('s save episode')
            if status:
                lines.append(status)
            draw_overlay(frame, lines)
            cv2.imshow('PickPlaceCan', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('r'):
                status = 'discarded episode'
                obs, model_xml, current_state, states, actions = reset_episode(env, device)
                continue
            if args.mode == 'collect' and key == ord('s'):
                if save_episode(data_group, demo_index, model_xml, states, actions):
                    print(f'Saved demo_{demo_index} ({len(actions)} steps, success={success})')
                    demo_index += 1
                    h5_file.flush()
                    status = 'saved episode'
                else:
                    status = 'nothing to save'
                obs, model_xml, current_state, states, actions = reset_episode(env, device)
                continue

            cycle_start = time.monotonic()
            action, grasp = input2action(device=device, robot=env.robots[0], active_arm='right',
                                         env_configuration=getattr(env, 'env_configuration', None))
            if action is None:
                status = 'reset from device'
                obs, model_xml, current_state, states, actions = reset_episode(env, device)
                continue

            if args.mode == 'collect':
                states.append(current_state.copy())
                actions.append(np.asarray(action, dtype=np.float32))
            obs, reward, _, _ = env.step(action)
            current_state = np.asarray(env.sim.get_state().flatten())
            status = f'reward={reward:.3f} grasp={grasp}'
            elapsed = time.monotonic() - cycle_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        device.close()
        env.close()
        if h5_file is not None:
            h5_file.close()
        cv2.destroyAllWindows()
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['teleop', 'collect'], default='teleop')
    parser.add_argument('--output', default='data/robomimic/datasets/can/custom/demo.hdf5')
    parser.add_argument('--robot', default='Panda')
    parser.add_argument('--env', default='PickPlaceCan')
    parser.add_argument('--camera', default='agentview')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--control-freq', type=int, default=20)
    parser.add_argument('--vendor-id', type=int, default=9583)
    parser.add_argument('--product-id', type=int, default=50741)
    parser.add_argument('--pos-sensitivity', type=float, default=0.005)
    parser.add_argument('--rot-sensitivity', type=float, default=0.001)
    parser.add_argument('--controller-kp', type=float, default=300.0)
    parser.add_argument('--controller-damping', type=float, default=2.5)
    parser.add_argument('--free-rotation', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
