from typing import List, Optional

import torch

from state_objects.composite_body import CompositeBody
from state_objects.primitive_shapes import Cylinder, SphereState, HollowCylinder
from state_objects.rigid_object import RigidBody
from utilities import torch_quaternion
from utilities.tensor_utils import tensorify, zeros


class TensegrityRod(CompositeBody):

    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sphere_radius: torch.Tensor,
                 sphere_mass: torch.Tensor,
                 motor_offset: torch.Tensor,
                 motor_length: torch.Tensor,
                 motor_radius: torch.Tensor,
                 motor_mass: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List[str],
                 quat: Optional[torch.Tensor] = None,
                 split_length: Optional[float] = None):
        prin_axis = end_pts[1] - end_pts[0]
        self.length = prin_axis.norm(dim=1, keepdim=True)
        prin_axis /= self.length

        rods = self._init_inner_rods(name,
                                     mass,
                                     end_pts,
                                     linear_vel,
                                     ang_vel,
                                     radius,
                                     split_length)
        endcaps = self._init_endcaps(name,
                                     end_pts,
                                     linear_vel,
                                     ang_vel,
                                     sphere_radius,
                                     sphere_mass,
                                     prin_axis,
                                     quat)
        motors = self._init_motors(name,
                                   (end_pts[1] + end_pts[0]) / 2.,
                                   ang_vel,
                                   linear_vel,
                                   motor_length,
                                   motor_mass,
                                   motor_offset,
                                   motor_radius,
                                   prin_axis,
                                   radius)

        rigid_bodies = rods + endcaps + motors
        self.num_rods = len(rods)
        self.motor_length = motor_length
        self.motor_offset = motor_offset
        self.sphere_radius = sphere_radius
        self.end_pts = end_pts

        if quat is None:
            quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin_axis)

        super().__init__(name,
                         linear_vel,
                         ang_vel,
                         quat,
                         rigid_bodies,
                         sites)

        self.body_verts, self.sphere_idx0, self.sphere_idx1 = (
            self._init_body_verts())

    def to(self, device):
        super().to(device)

        self.motor_length = self.motor_length.to(device)
        self.motor_offset = self.motor_offset.to(device)
        self.sphere_radius = self.sphere_radius.to(device)
        self.length = self.length.to(device)
        self.end_pts = [e.to(device) for e in self.end_pts]

        self.body_verts = self.body_verts.to(device)

        return self

    @classmethod
    def init_from_cfg(cls, cfg):
        cfg_copy = {k: v for k, v in cfg.items()}

        end_pts = tensorify(cfg['end_pts'], reshape=(2, 3, 1))
        cfg_copy['end_pts'] = [end_pts[:1], end_pts[1:]]

        cfg_copy['radius'] = tensorify(cfg['radius'], reshape=(1, 1, 1))
        cfg_copy['mass'] = tensorify(cfg['mass'], reshape=(1, 1, 1))
        cfg_copy['sphere_radius'] = tensorify(cfg['sphere_radius'], reshape=(1, 1, 1))
        cfg_copy['sphere_mass'] = tensorify(cfg['sphere_mass'], reshape=(1, 1, 1))
        cfg_copy['motor_offset'] = tensorify(cfg['motor_offset'], reshape=(1, 1, 1))
        cfg_copy['motor_length'] = tensorify(cfg['motor_length'], reshape=(1, 1, 1))
        cfg_copy['motor_radius'] = tensorify(cfg['motor_radius'], reshape=(1, 1, 1))
        cfg_copy['motor_mass'] = tensorify(cfg['motor_mass'], reshape=(1, 1, 1))
        cfg_copy['linear_vel'] = tensorify(cfg['linear_vel'], reshape=(1, 3, 1))
        cfg_copy['ang_vel'] = tensorify(cfg['ang_vel'], reshape=(1, 3, 1))

        return cls(**cfg_copy)

    def _init_body_verts(self):
        body_verts = []
        sphere0_idx, sphere1_idx = -1, -1
        inv_quat = torch_quaternion.inverse_unit_quat(self.quat)
        for j, body in enumerate(self.rigid_bodies.values()):
            world_vert = self.rigid_bodies[body.name].pos
            body_vert = torch_quaternion.rotate_vec_quat(
                inv_quat,
                world_vert - self.pos
            )
            body_verts.append(body_vert)

            if "sphere0" in body.name:
                sphere0_idx = j
            elif "sphere1" in body.name:
                sphere1_idx = j

        body_verts = torch.vstack(body_verts)

        return body_verts, sphere0_idx, sphere1_idx

    def _init_motors(self,
                     name,
                     pos,
                     ang_vel,
                     linear_vel,
                     motor_length,
                     motor_mass,
                     motor_offset,
                     motor_radius,
                     prin_axis,
                     radius):
        motor_e1_dist = (motor_length / 2 + motor_offset) * prin_axis
        motor_e2_dist = (-motor_length / 2 + motor_offset) * prin_axis
        ang_vel_comp = torch.cross(ang_vel, motor_offset * prin_axis)
        motor0 = HollowCylinder(f'{name}_motor0',
                                [pos - motor_e1_dist, pos - motor_e2_dist],
                                linear_vel - ang_vel_comp,
                                ang_vel.clone(),
                                motor_radius,
                                radius,
                                motor_mass,
                                {})
        motor1 = HollowCylinder(f'{name}_motor1',
                                [pos + motor_e2_dist, pos + motor_e1_dist],
                                linear_vel + ang_vel_comp,
                                ang_vel.clone(),
                                motor_radius,
                                radius,
                                motor_mass,
                                {})
        return [motor0, motor1]

    def _init_endcaps(self,
                      name,
                      end_pts,
                      linear_vel,
                      ang_vel,
                      sphere_radius,
                      sphere_mass,
                      prin_axis,
                      quat):
        endcap0 = SphereState(name + "_sphere0",
                              end_pts[0],
                              linear_vel.clone(),
                              ang_vel.clone(),
                              sphere_radius,
                              sphere_mass,
                              prin_axis,
                              {},
                              quat)
        endcap1 = SphereState(name + "_sphere1",
                              end_pts[1],
                              linear_vel.clone(),
                              ang_vel.clone(),
                              sphere_radius,
                              sphere_mass,
                              prin_axis,
                              {},
                              quat)

        return [endcap0, endcap1]

    def _init_inner_rods(self,
                         name,
                         mass,
                         end_pts,
                         lin_vel,
                         ang_vel,
                         radius,
                         split_length):

        if split_length:
            rod_prin_axis = end_pts[1] - end_pts[0]
            rod_length = rod_prin_axis.norm(keepdim=True)
            rod_prin_axis /= rod_length

            inner_length = split_length
            num_rods = int(rod_length / inner_length)
            outer_length = (rod_length - num_rods * inner_length) / 2.0 + inner_length
            offsets = torch.tensor(([0, outer_length]
                                    + [inner_length] * (num_rods - 2)
                                    + [outer_length]))
            offsets1 = torch.cumsum(offsets[:-1], dim=0)
            offsets2 = torch.cumsum(offsets[1:], dim=0)

            rods = []
            for i in range(num_rods):
                offset1, offset2 = offsets1[i], offsets2[i]
                rod_end_pts = torch.concat([
                    end_pts[0] + offset1 * rod_prin_axis,
                    end_pts[0] + offset2 * rod_prin_axis
                ], dim=-1)

                rod_mass = mass * (offset2 - offset1) / rod_length

                rod = Cylinder(name + f"_rod{i}",
                               rod_end_pts,
                               lin_vel,
                               ang_vel,
                               radius,
                               rod_mass,
                               {})
                rods.append(rod)
        else:
            rods = [
                Cylinder(name + "_rod",
                         end_pts,
                         lin_vel,
                         ang_vel,
                         radius,
                         mass,
                         {})
            ]

        return rods

    def update_state_by_endpts(self, end_pts, lin_vel, ang_vel):
        curr_prin = end_pts[1] - end_pts[0]
        curr_prin = curr_prin / curr_prin.norm(dim=1, keepdim=True)

        pos = (end_pts[0] + end_pts[1]) / 2.0
        quat = torch_quaternion.compute_quat_btwn_z_and_vec(curr_prin)

        self.update_state(pos, lin_vel, quat, ang_vel)

    def get_template_graph(self):
        template_graph = [
            (f"{self.name}_rod0", f"{self.name}_sphere0"),
            (f"{self.name}_sphere0", f"{self.name}_rod0")
        ]

        for i in range(self.num_rods - 1):
            template_graph.append((f"{self.name}_rod{i}",
                                   f"{self.name}_rod{i + 1}"))
            template_graph.append((f"{self.name}_rod{i + 1}",
                                   f"{self.name}_rod{i}"))

        template_graph.append((f"{self.name}_rod{self.num_rods - 1}",
                               f"{self.name}_sphere1"))
        template_graph.append((f"{self.name}_sphere1",
                               f"{self.name}_rod{self.num_rods - 1}"))

        motor0 = self.rigid_bodies[f"{self.name}_motor0"]
        motor1 = self.rigid_bodies[f"{self.name}_motor1"]

        for i in range(self.num_rods):
            rod_name = f"{self.name}_rod{i}"
            rod = self.rigid_bodies[rod_name]

            for motor in [motor0, motor1]:
                if self._overlap_rods(rod, motor):
                    template_graph.append((rod.name, motor.name))
                    template_graph.append((motor.name, rod.name))

        return template_graph

    def _overlap_rods(self, rod1, rod2):
        # assuming parallel/concentric
        def rod_inside(rod_a, rod_b):
            prin_axis = rod_a.get_principal_axis()
            for end_pt in rod_b.end_pts:
                rel_vec = end_pt - rod_a.end_pts[0]
                length = torch.linalg.vecdot(prin_axis, rel_vec, dim=1)
                if 0 <= length <= rod_a.length:
                    return True
            return False

        return rod_inside(rod1, rod2) or rod_inside(rod2, rod1)

    def update_state(self, pos, linear_vel, quat, ang_vel):
        super().update_state(pos, linear_vel, quat, ang_vel)

        prin_axis = Cylinder.compute_principal_axis(quat)
        self.end_pts = Cylinder.compute_end_pts_from_state(
            self.state,
            prin_axis,
            self.length
        )

    def update_sites(self, site, pos):
        self.sites[site] = pos

    def _particle_constraints(self, p0, p1, p2, d0, d1, start_mid_end='mid'):
        val = None
        if start_mid_end == 'start':
            p01 = (p1 - p0).norm(dim=1, keepdim=True)
            p02 = (p2 - p0).norm(dim=1, keepdim=True)
            val = p01 - d0 + p02 - d1
        elif start_mid_end == 'mid':
            p01 = (p1 - p0).norm(dim=1, keepdim=True)
            p12 = (p2 - p1).norm(dim=1, keepdim=True)
            val = p01 - d0 + p12 - d1
        elif start_mid_end == 'end':
            p02 = (p2 - p0).norm(dim=1, keepdim=True)
            p12 = (p2 - p1).norm(dim=1, keepdim=True)
            val = p02 - d0 + p12 - d1
        return val

    def rigid_body_constraints(self, node_pos):
        num_nodes = len(self.rigid_bodies_body_vecs)
        p0_idx = [num_nodes - 1] + list(range(num_nodes - 1))
        p1_idx = list(range(num_nodes))
        p2_idx = list(range(1, num_nodes)) + [0]

        node_pos = node_pos.unsqueeze(-1).transpose(1, 2).reshape(-1, num_nodes, 3).transpose(1, 2)
        p0 = node_pos[..., p0_idx]
        p1 = node_pos[..., p1_idx]
        p2 = node_pos[..., p2_idx]

        p01 = (p0 - p1).norm(dim=1, keepdim=True).transpose(1, 2)
        p12 = (p1 - p2).norm(dim=1, keepdim=True).transpose(1, 2)

        body_vecs = self.body_vecs_tensor.to(node_pos.device)
        d01 = (body_vecs[p0_idx] - body_vecs[p1_idx]).norm(dim=1, keepdim=True).transpose(0, 1)
        d12 = (body_vecs[p2_idx] - body_vecs[p1_idx]).norm(dim=1, keepdim=True).transpose(0, 1)
        val = p01 - d01 + p12 - d12

        return val

    def rigid_body_jacobian(self, node_pos):
        num_nodes = len(self.rigid_bodies_body_vecs)
        p0_idx = [num_nodes - 1] + list(range(num_nodes - 1))
        p1_idx = list(range(num_nodes))
        p2_idx = list(range(1, num_nodes)) + [0]

        node_pos = node_pos.unsqueeze(-1).transpose(1, 2).reshape(-1, num_nodes, 3).transpose(1, 2)
        p0 = node_pos[..., p0_idx]
        p1 = node_pos[..., p1_idx]
        p2 = node_pos[..., p2_idx]

        a, c = (p1 - p0).transpose(1, 2), (p2 - p1).transpose(1, 2)
        a = a / a.norm(dim=2, keepdim=True)
        c = c / c.norm(dim=2, keepdim=True)
        b = a - c

        batch_size = node_pos.shape[0]
        jacobian = zeros((batch_size, num_nodes, 3 * num_nodes),
                         ref_tensor=node_pos)

        row_idx = torch.arange(0, num_nodes).unsqueeze(1).repeat(1, 3)
        col_idx = torch.arange(0, 3 * num_nodes).reshape(num_nodes, -1)
        jacobian[:, row_idx, col_idx] = b
        jacobian[:, row_idx, col_idx[p0_idx]] = -a
        jacobian[:, row_idx, col_idx[p2_idx]] = c

        return jacobian

    def compute_rb_constraint_dv(self,
                                 node_pos,
                                 temp_vels,
                                 batch_size,
                                 dt,
                                 inv_mass_mat=None,
                                 jacob=None):
        if inv_mass_mat is None:
            inv_mass_mat = self.inv_mass_mat

        if jacob is None:
            jacob = self.rigid_body_jacobian(node_pos)

        jacob_t = jacob.transpose(1, 2)

        cons_err = self.rigid_body_constraints(node_pos).reshape(batch_size, -1, 1)
        bias = 0.2 * cons_err / dt

        v = jacob @ temp_vels.reshape(batch_size, -1, 1) + bias
        vv = torch.linalg.solve(jacob @ (inv_mass_mat @ jacob_t), v)
        dv = -inv_mass_mat @ (jacob_t @ vv)
        dv = dv.reshape(node_pos.shape)

        return dv

    @property
    def inv_mass_mat(self):
        inv_masses = self.inv_mass_vec.squeeze(-1).repeat(1, 3).flatten()
        return torch.diag(inv_masses).unsqueeze(0)

    @property
    def inv_mass_vec(self):
        masses = [body.mass for body in self.rigid_bodies.values()]
        inv_masses = 1. / torch.hstack(masses)

        return inv_masses


class TensegrityRodVN(TensegrityRod):

    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sphere_radius: torch.Tensor,
                 sphere_mass: torch.Tensor,
                 motor_offset: torch.Tensor,
                 motor_length: torch.Tensor,
                 motor_radius: torch.Tensor,
                 motor_mass: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List[str],
                 quat: Optional[torch.Tensor] = None,
                 split_length: Optional[float] = None):
        super().__init__(name,
                         end_pts,
                         radius,
                         mass,
                         sphere_radius,
                         sphere_mass,
                         motor_offset,
                         motor_length,
                         motor_radius,
                         motor_mass,
                         linear_vel,
                         ang_vel,
                         sites,
                         quat,
                         split_length)

        virtual_body = RigidBody(name + "_vn",
                                 self.mass,
                                 self.I_body,
                                 self.pos,
                                 self.quat,
                                 self.linear_vel,
                                 self.ang_vel,
                                 self.sites)

        vn_body_pos = torch.zeros((1, 3, 1), dtype=self.dtype)

        self._rigid_bodies[virtual_body.name] = virtual_body
        self._rigid_bodies_body_vecs[virtual_body.name] = vn_body_pos

        self.body_verts = torch.vstack([self.body_verts, vn_body_pos])
        self.body_vecs_tensor = torch.vstack([self.body_vecs_tensor, vn_body_pos])
        self.vn_idx = len(self.rigid_bodies)

