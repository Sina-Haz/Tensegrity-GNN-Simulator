"""
In this file we want to accomplish following:

1. Write a concise data structure holding all necessary "State" info we need per simulation step
 - Store pos, quat, v, w of each rod
 - Store "body verts", aka a list of points in local frame for each rod -> get mapped to world frame
 - Store "num_nodes" -> this is used to create the body mask
 - Store inverse mass and inertia properties (per rod?)
 - Sphere radius (and perhaps the idx's of body verts which map to endcap0, endcap1)
 - Cable properties such as stiffness, damping, rest len, act-len, AND connectivity (sites?)

2. Write functions that can map this data structure to a graph data structure suitable for feeding into the GNN
    - The code for this is now in transofrms.py

3. Write functions that can map the output graph "node-level" data back to "State"
    - can map node level positions to rod positions, but doing that for ang velocity, quat, etc. needs to be done as well
    - I think rest length and motor speed would only be updated based on control input
"""

import jax.numpy as np
import torch
import equinox as eqx
import sys
import os

# Get the absolute path of the codebase root (one level above `gnn_physics`)
codebase_root = os.path.abspath("tensegrity_gnn_compiled")

from robots.tensegrity import TensegrityRobotGNN
from gnn_physics.data_processors.batch_tensegrity_data_processor import BatchTensegrityDataProcessor
from state_objects.cables import ActuatedCable



# Add it to sys.path
sys.path.insert(0, codebase_root)


# Helper method for conversion from torch.Tensor -> jax.numpy.ndarray
def to_np(tensor: torch.tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return np.array(tensor.detach().cpu().numpy())
    elif isinstance(tensor, np.ndarray): return tensor
    else:
        raise Exception # input was neither tensor nor np array



class Robot(eqx.Module):
    P: np.ndarray # shape (n_rods, 3)
    Q: np.ndarray # shape (n_rods, 4)
    V: np.ndarray # shape (n_rods, 3)
    W: np.ndarray # shape (n_rods, 3)

    # Data for node gen. and features
    local_nodes: np.ndarray # shape (n_rods, n_verts, 3), use this to get n_nodes
    inv_I: np.ndarray # shape (n_rods * n_verts, 3) -> iterates over every r.b. and stacks diagonals
    inv_M: np.ndarray # shape (n_rods * n_verts, 1)
    endcap_R: float = eqx.field(static=True) # const. for all spheres (endcaps)
    frame_idx_1: int = eqx.field(static=True) # an index for flattened node positions needed for computing frame-wise features
    frame_idx_2: int = eqx.field(static=True) # for more see frame_features in transforms

    # Data for cable edge gen. and features
    ks: np.ndarray # shape (2*n_cables, )
    kd: np.ndarray # shape (2*n_cables, )
    rest_len: np.ndarray # shape (2*n_cables, )

    # Motor and actuation necessary data
    w_t: np.ndarray # shape (n_cables, ) motor angular velocity state at time t (0 for un-actuated cables)
    motor_speed: np.ndarray # (n_cables, ) motor speed (0-1) is 0 for non-actuated cables
    winch_r: float = eqx.field(static=True) # uniform for all actuated cables
    max_len: float = eqx.field(static=True) # uniform for all actuated cables
    min_len: float = eqx.field(static=True) # uniform for all actuated cables

    body_edges: np.ndarray # shape (2, num_body_edges), body_edge[:, i] should give indices to 2 nodes
    cable_edges: np.ndarray # shape (2, num_cable_edges), cable_edge[:, i] should give indices to 2 nodes
    contact_edges: np.ndarray # shape (2, num_sphere_nodes), contact_edge[:, i] -> node idx of a sphere + ground node idx, need to filter this

    @classmethod
    def TensegrityToRobot(cls, tensegrity: TensegrityRobotGNN):
        """
        In this method we extract necessary data from Tensegrity and move it over to JAX data structure
        """
        n_rods, rods = len(tensegrity.rods), list(tensegrity.rods.values())
        # Write in state info and remove extra dimensions
        P, Q, W, V = to_np(tensegrity.pos).reshape(n_rods, 3), to_np(tensegrity.quat).reshape(n_rods, 4),\
              to_np(tensegrity.ang_vel).reshape(n_rods, 3), to_np(tensegrity.linear_vel).reshape(n_rods, 3)
        
        loc_nodes = to_np(tensegrity.body_verts).squeeze().reshape(len(tensegrity.rods), -1, 3)
        inv_I, inv_M = to_np(tensegrity.inv_inertia).squeeze(), to_np(tensegrity.inv_mass).squeeze(-1)
        endcap_R = float(tensegrity.sphere_radius)
        frame_idx1, frame_idx2 = rods[0].sphere_idx0, rods[-1].sphere_idx1 
        frame_idx2 += sum([r.body_verts.shape[0] for r in rods[:-1]]) # added this s.t. idx accts for the fact that it's in the last rod

        # Get cable properties in graph form (i.e. in shape 2*n_cables, ...), rest_len is stateful
        ks, kd, rest_len = to_np(tensegrity.cable_stiffness).squeeze(), to_np(tensegrity.cable_damping).squeeze(), \
            to_np(torch.tensor([r for c in tensegrity.cables.values() for r in [c.rest_length, c.rest_length]])).squeeze()
        
        # Get motor values: w_t is "stateful"
        w_t = np.array([c.motor.motor_state.omega_t.detach().item() if isinstance(c, ActuatedCable) else 0 for c in tensegrity.cables.values() ])
        motor_speed = np.array([c.motor.speed if isinstance(c, ActuatedCable) else 0 for c in tensegrity.cables.values()])
        winch_r = next((c.winch_r.detach().item() for c in tensegrity.cables.values() if isinstance(c, ActuatedCable)), 0)
        max_len = next((c.max_length for c in tensegrity.cables.values() if isinstance(c, ActuatedCable)), 0)
        min_len = next((c.min_length for c in tensegrity.cables.values() if isinstance(c, ActuatedCable)), 0)

        # Instantiate data processor so we can store the edges
        dp = BatchTensegrityDataProcessor(tensegrity)
        n_verts = loc_nodes.shape[1] # store number of nodes per rod

        # Use the tensegrity methods to store body, cable, and contact edge indices (contact will be filtered dynamically)
        body_edges = np.array(dp._body_edge_index(), dtype=np.int32)
        cable_edges = np.array(dp._get_cable_edge_idxs(), dtype=np.int32)
        # ground idx = n_rods * n_verts b/c of zero indexing, contact_edges = sphere node indices to grnd node idx
        contact_edges = np.array(dp._contact_edge_index(tensegrity.get_contact_nodes(), n_rods * n_verts))
        
        # Return robot instance
        return cls(
            P=P, Q=Q, V=V, W=W,
            local_nodes=loc_nodes, inv_I=inv_I, inv_M=inv_M, endcap_R=endcap_R, frame_idx_1=frame_idx1, frame_idx_2=frame_idx2,
            ks=ks, kd=kd, rest_len=rest_len,
            w_t=w_t, motor_speed=motor_speed, winch_r=winch_r, max_len=max_len, min_len=min_len,
            body_edges=body_edges, cable_edges=cable_edges, contact_edges=contact_edges
        )
    

    def updateState(self, P, Q, V, W, rest_len, w_t):
        """
        Return a new robot with updated state (since fields are all immutable)

        inputs can be torch tensors or jax arrays, and can have weird dim
        """
        # Reshape state vars if necessary
        n_rods = len(self.P)
        P, Q, W, V = to_np(P).reshape(n_rods, 3), to_np(Q).reshape(n_rods, 4),\
              to_np(W).reshape(n_rods, 3), to_np(V).reshape(n_rods, 3)
        
        # Reshape and update cable and motor states:
        n_cables = len(self.motor_speed)

        if rest_len.shape == (n_cables, ):
            rest_len = np.repeat(rest_len, 2)
        assert rest_len.shape == (2*n_cables, )

        assert w_t.shape == (n_cables, )
        
        return eqx.tree_at(lambda r: (r.P, r.Q, r.V, r.W, r.rest_len, r.w_t),
                           self,
                           (P, Q, V, W, rest_len, w_t))
    








