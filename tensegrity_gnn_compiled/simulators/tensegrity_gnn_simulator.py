from typing import Union

from torch_geometric.data import Data as Graph

from gnn_physics.data_processors.batch_tensegrity_data_processor import *
from gnn_physics.gnn import *
from robots.tensegrity import TensegrityRobotGNN
from state_objects.base_state_object import BaseStateObject


class LearnedSimulator(BaseStateObject):

    def __init__(
            self,
            node_types: Dict[str, int],
            edge_types: Dict[str, int],
            n_out: int,
            latent_dim: int,
            nmessage_passing_steps: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            data_processor: AbstractRobotGraphDataProcessor,
            processor_shared_weights=False):
        super().__init__('learned simulator')

        self.node_types = node_types
        self.edge_types = edge_types

        self.data_processor = data_processor

        # Initialize the EncodeProcessDecode
        self._encode_process_decode = self.build_gnn(
            node_types=node_types,
            edge_types=edge_types,
            n_out=n_out,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            processor_shared_weights=processor_shared_weights,
        )

    def update_state(self, next_state: torch.Tensor) -> None:
        next_state_ = next_state.reshape(-1, 13, 1)
        pos, quat = next_state_[:, :3], next_state_[:, 3:7]
        lin_vel, ang_vel = next_state_[:, 7:10], next_state_[:, 10:]

        self.robot.update_state(
            pos.reshape(next_state.shape[0], -1, 1),
            lin_vel.reshape(next_state.shape[0], -1, 1),
            quat.reshape(next_state.shape[0], -1, 1),
            ang_vel.reshape(next_state.shape[0], -1, 1),
        )

    def apply_control(self, control_signals, dt):
       pass

    def build_gnn(self, **kwargs):
        return EncodeProcessDecode(
            node_types=kwargs['node_types'],
            edge_types=kwargs['edge_types'],
            n_out=kwargs['n_out'],
            latent_dim=kwargs['latent_dim'],
            nmessage_passing_steps=kwargs['nmessage_passing_steps'],
            nmlp_layers=kwargs['nmlp_layers'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'],
            processor_shared_weights=kwargs['processor_shared_weights'],
        )

    @property
    def robot(self):
        return self.data_processor.robot

    def to(self, device):
        super().to(device)
        self._encode_process_decode = self._encode_process_decode.to(device)
        self.data_processor = self.data_processor.to(device)

        return self

    def forward(self,
                curr_state: torch.Tensor,
                dt: Union[torch.Tensor, float],
                control_signals: torch.Tensor):
        next_state = self.step(curr_state, dt, control_signals)
        return next_state

    def step(self,
             curr_state: torch.Tensor,
             dt: Union[torch.Tensor, float],
             control_signals: torch.Tensor):
        pass


class TensegrityGNNSimulator(LearnedSimulator):

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 processor_shared_weights=False,
                 dt=0.01,
                 tensegrity_cfg=None,
                 robot=None,
                 n_hist=1):
        assert robot is not None or tensegrity_cfg is not None

        self.prev_states = None
        self.prev_act_lens = None
        self.ctrls_hist = None
        self.curr_graph = None

        if robot is None:
            robot = TensegrityRobotGNN(tensegrity_cfg)

        data_processor = BatchTensegrityDataProcessor(robot, dt=dt, num_hist=n_hist)

        node_types = {k: sum(v.values()) for k, v in data_processor.hier_node_feat_dict.items()}
        edge_types = {k: sum(v.values()) for k, v in data_processor.hier_edge_feat_dict.items()}

        super().__init__(node_types,
                         edge_types,
                         n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         data_processor,
                         processor_shared_weights)

        if self.dtype == torch.float64:
            self._encode_process_decode = self._encode_process_decode.double()

    def reset(self):
        self.prev_states = None
        self.prev_act_lens = None
        self.curr_graph = None
        self.ctrls_hist = None

    def build_gnn(self, **kwargs):
        return EncodeProcessDecode(
            node_types=kwargs['node_types'],
            edge_types=kwargs['edge_types'],
            n_out=kwargs['n_out'],
            latent_dim=kwargs['latent_dim'],
            nmessage_passing_steps=kwargs['nmessage_passing_steps'],
            nmlp_layers=kwargs['nmlp_layers'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'],
            processor_shared_weights=kwargs['processor_shared_weights'],
            # n_hist=1 + (len(self.prev_states) if self.prev_states else 0)
        )

    def apply_control(self, control_signals, dt):
        if control_signals is None:
            return

        # Seems like the shape of control signals is (batch_size, num_cables, ...)
        # Here he creates a dict where each cable gets ctrl input of shape (bs, 1, 1)
        if isinstance(control_signals, torch.Tensor):
            control_signals = {
                f'cable_{i}': control_signals[:, i: i + 1, None]
                for i in range(control_signals.shape[1])
            }
        elif isinstance(control_signals, list):
            control_signals = {
                f'cable_{i}':
                    ctrl.reshape(-1, 1, 1)
                    if isinstance(ctrl, torch.Tensor)
                    else torch.tensor(ctrl, dtype=self.dtype).reshape(-1, 1, 1)
                for i, ctrl in enumerate(control_signals)
            }

        for name, control in control_signals.items():
            if not isinstance(control, torch.Tensor):
                control = torch.tensor(
                    control,
                    dtype=self.dtype,
                    device=self.device
                ).reshape(-1, 1)

            measure_name = self.robot.cable_map[name]
            measure_cable = self.robot.cables[measure_name]
            cable = self.robot.cables[name]

            # For each control to a cable -> need current length, current rest length
            curr_length, _ = self.robot.compute_cable_length(measure_cable)
            cable.update_rest_length(control, curr_length, dt)

    def process_gnn(self, state) -> Graph:
        # Store states needed for processing
        states = [state]
        if self.prev_states is not None:
            states = self.prev_states + states

        # This code concatenates previous actuation lengths to the current actuated cables' data
        if self.prev_act_lens is not None:
            for i, c in enumerate(self.robot.actuated_cables.values()):
                c.actuation_length = torch.concat([
                    self.prev_act_lens[:, i: i + 1], c.actuation_length
                ], dim=2)

        # build a graph from a batch of states and history of controls
        graph = self.data_processor.batch_state_to_graph(states, ctrls=self.ctrls_hist)
        graph = self._encode_process_decode(graph)

        normalizer = self.data_processor.normalizers['dv']

        graph['p_dv'] = normalizer.inverse(graph['decode_output'])
        graph['p_vel'] = graph.vel + graph.p_dv
        graph['p_pos'] = graph.pos + self.data_processor.dt * graph.p_vel

        return graph

    
    def step(self, state, dt, ctrls=None):
        self.update_state(state)
        self.apply_controls(ctrls)

        graph = self.process_gnn(state)

        body_mask = graph.body_mask.flatten()
        next_state = self.data_processor.node2pose(
            graph.p_pos[body_mask],
            graph.pos[body_mask],
            self.robot.num_nodes_per_rod
        )

        return next_state
