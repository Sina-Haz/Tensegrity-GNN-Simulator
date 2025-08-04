import json
import random
from copy import deepcopy
from pathlib import Path

import torch
import tqdm

# from mujoco_visualizer_utils.mujoco_visualizer import MuJoCoVisualizer
# from simulators.tensegrity_simulator import Tensegrity5dRobotSimulator
from simulators.tensegrity_gnn_simulator import TensegrityHybridGNNSimulator
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE


def rollout_n_steps(simulator,
                    ctrls,
                    dt,
                    start_state):
    poses = []

    # Get the current state (pose + velocity)
    curr_state = start_state \
        if start_state is not None \
        else simulator.get_curr_state()
    
    # We reshape the current state into separate state components of 13x1 arrays
    # Extract first 7 elements of the state (pose) then reshape back to ensure same batch dimension
    pose = curr_state.reshape(-1, 13, 1)[:, :7].reshape(curr_state.shape[0], -1, 1)
    poses.append(pose)

    
    for i, ctrl in enumerate(tqdm.tqdm(ctrls)):
        with torch.no_grad():
            curr_state, _ = simulator.step(
                curr_state,
                dt,
                control_signals=ctrl
            )

        pose = curr_state.reshape(-1, 13, 1)[:, :7].reshape(curr_state.shape[0], -1, 1)
        poses.append(pose)

    return poses


def rollout_by_ctrls(simulator,
                     ctrls,
                     dt,
                     start_state,
                     gt_data):
    time = 0.0
    frames = []

    curr_state = start_state \
        if start_state is not None \
        else simulator.get_curr_state()
    pose = curr_state.reshape(-1, 13, 1)[:, :7].flatten().numpy()
    frames.append({"time": time, "pos": pose.tolist()})

    for i, ctrl in enumerate(tqdm.tqdm(ctrls)):
        with torch.no_grad():
            if (i + 1) % 200 == 0:
                prev_end_pts = torch.tensor(gt_data[i - 1]['end_pts']).reshape(-1, 6, 1)
                prev_pos = (prev_end_pts[:, :3] + prev_end_pts[:, 3:]) / 2.
                prev_quat = torch_quaternion.compute_quat_btwn_z_and_vec(
                    prev_end_pts[:, 3:] - prev_end_pts[:, :3]
                )

                end_pts = torch.tensor(gt_data[i]['end_pts']).reshape(-1, 6, 1)
                pos = (end_pts[:, :3] + end_pts[:, 3:]) / 2.
                quat = torch_quaternion.compute_quat_btwn_z_and_vec(
                    end_pts[:, 3:] - end_pts[:, :3]
                )

                lin_vels = (pos - prev_pos) / dt
                ang_vels = torch_quaternion.compute_ang_vel_quat(prev_quat, quat, dt)

                curr_state = torch.hstack([pos, quat, lin_vels, ang_vels]).reshape(1, -1, 1)

            curr_state, _ = simulator.step(
                curr_state,
                dt,
                control_signals=ctrl
            )

        # time += dt
        pose = curr_state.reshape(-1, 13, 1)[:, :7].flatten().numpy()
        frames.append({'time': time, 'pos': pose.tolist()})

    return frames


def evaluate(simulator,
             gt_data,
             ctrls,
             init_rest_lengths,
             init_motor_speeds,
             dt):
    cables = simulator.actuated_cables.values()
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - init_rest_lengths[i]
        c.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1)

    pos, quat = gt_data[0]['pos'], gt_data[0]['quat']
    linvel, angvel = gt_data[0]['linvel'], gt_data[0]['angvel']

    start_state = torch.tensor(
        pos + quat + linvel + angvel,
        dtype=DEFAULT_DTYPE
    ).reshape(1, 13, 1)

    rollout_poses = rollout_by_ctrls(
        simulator,
        ctrls,
        dt,
        start_state
    )

    com_errs, rot_errs, pen_errs = [], [], []
    for i in range(1, len(gt_data)):
        gt_pos = torch.tensor(
            gt_data[i]['pos'],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 3, 1)

        gt_quat = torch.tensor(
            gt_data[i]['quat'],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 4, 1)

        pred_pos = rollout_poses[i]['pose'][:, :3]
        pred_quat = rollout_poses[i]['pose'][:, 3:7]

        com_mse = ((gt_pos - pred_pos) ** 2).mean()
        ang_err = torch_quaternion.compute_angle_btwn_quats(gt_quat, pred_quat)

        gt_pen = torch.clamp_max(gt_pos[:, 2], 0.0)
        pred_pen = torch.clamp_max(pred_pos[:, 2], 0.0)
        pen_err = torch.clamp_min(gt_pen - pred_pen, 0.0)

        com_errs.append(com_mse.item())
        rot_errs.append(ang_err.item())
        pen_errs.append(pen_err.item())

    avg_com_err = sum(com_errs) / len(com_errs)
    avg_rot_err = sum(rot_errs) / len(rot_errs)
    avg_pen_err = sum(pen_errs) / len(pen_errs)

    return avg_com_err, avg_rot_err, avg_pen_err


def compute_end_pts_from_state(rod_pos_state, principal_axis, rod_length):
    """
    :param rod_pos_state: (x, y, z, quat.w, quat.x, quat.y, quat.z)
    :param principal_axis: tensor of vector(s)
    :param rod_length: length of rod
    :return: ((x1, y1, z1), (x2, y2, z2))
    """
    # Get position
    pos = rod_pos_state[:, :3, ...]

    # Compute half-length vector from principal axis
    half_length_vec = rod_length * principal_axis / 2

    # End points are +/- of half-length vector from COM
    end_pt1 = pos - half_length_vec
    end_pt2 = pos + half_length_vec

    return [end_pt1, end_pt2]


def batch_compute_end_pts(sim, batch_state: torch.Tensor):
    """
    Compute end pts for entire batch

    :param batch_state: batch of states
    :return: list of endpts
    """
    end_pts = []
    for i, rod in enumerate(sim.rigid_bodies.values()):
        state = batch_state[:, i * 7: i * 7 + 7]
        principal_axis = torch_quaternion.quat_as_rot_mat(state[:, 3:7])[..., 2:]
        end_pts.extend(compute_end_pts_from_state(state, principal_axis, rod.length))

    return torch.hstack(end_pts)


if __name__ == '__main__':
    torch.set_num_threads(1)

    base_path = Path("../../tensegrity/data_sets/tensegrity_real_datasets"
                     "/synthetic/mjc_synthetic_5d_0.01/val/R2S2Rcw_19/")
    model_path = Path(base_path, "../../../../mjc_syn_models/5d_new_model/")
    sim = torch.load(Path(model_path, "2_steps_best_rollout_model.pt"), map_location='cpu')
    sim.to("cuda")
    sim.eval()

    vis_gt_data = json.load(open(base_path / "processed_data.json"))
    extra_states_json = json.load(open(base_path / "extra_state_data.json"))

    ctrls = [e['controls'] for e in extra_states_json]

    for i, spring in enumerate(sim.robot.actuated_cables.values()):
        rest_length = extra_states_json[0]['rest_lengths'][i]
        act_length = spring._rest_length - rest_length
        spring.motor.speed = torch.tensor(0.8, dtype=DEFAULT_DTYPE).reshape(1, 1, 1)
        spring.actuation_length = act_length
        spring.min_length = 0.5
        spring.max_length = 2.4

    rod_end_pts = [torch.tensor(e, dtype=DEFAULT_DTYPE).reshape(1, -1, 1) for e in vis_gt_data[0]['end_pts']]
    pos = [0.5 * (rod_end_pts[2 * i + 1] + rod_end_pts[2 * i]) for i in range(len(rod_end_pts) // 2)]
    prins = [rod_end_pts[2 * i + 1] - rod_end_pts[2 * i] for i in range(len(rod_end_pts) // 2)]
    quats = [torch_quaternion.compute_quat_btwn_z_and_vec(p) for p in prins]
    linvels = [torch.zeros((1, 3, 1), dtype=DEFAULT_DTYPE) for i in range(3)]
    angvels = [torch.zeros((1, 3, 1), dtype=DEFAULT_DTYPE) for i in range(3)]
    state = torch.hstack([
        torch.hstack([pos[i], quats[i], linvels[i], angvels[i]])
        for i in range(len(pos))
    ])
    sim.update_state(state)

    frames = rollout_by_ctrls(sim, ctrls[:], 0.01, None, vis_gt_data)

    if len(vis_gt_data) >= len(frames):
        last_frame = deepcopy(frames[-1])
        for i, data in enumerate(vis_gt_data):
            t = data['time']
            pos = data['pos']
            quat = data['quat']
            pose = [p for j in range(len(pos) // 3)
                    for p in (pos[j * 3: (j + 1) * 3] + quat[j * 4: (j + 1) * 4])]
            # pose = data['pos']

            if i < len(frames):
                frames[i]['pos'] += pose
            else:
                frames.append({
                    "time": t,
                    'pos': last_frame['pos'][:7 * len(pos) // 3] + pose
                })
    else:
        last_frame = deepcopy(vis_gt_data[-1])
        for i in range(len(frames)):

            if i < len(vis_gt_data):
                pos = vis_gt_data[i]['pos']
                quat = vis_gt_data[i]['quat']
                pose = [p for j in range(len(pos) // 3)
                        for p in (pos[j * 3: (j + 1) * 3] + quat[j * 4: (j + 1) * 4])]
                # pose = vis_gt_data[i]['pos']

                frames[i]['pos'] += pose
            else:
                pos = last_frame['pos']
                quat = last_frame['quat']
                pose = [p for j in range(len(pos) // 3) for p in
                        (pos[j * 3: (j + 1) * 3] + quat[j * 4: (j + 1) * 4])]
                # pose = last_frame['pos']
                frames[i]['pos'] += pose

    vis = MuJoCoVisualizer()
    vis.set_xml_path(Path("mujoco_physics_engine/xml_models/3prism_real_upscaled_vis_w_gt.xml"))
    vis.set_camera("camera")
    vis.data = frames
    vis.visualize(Path(model_path, f"{base_path.name}.mp4"), 0.01)

