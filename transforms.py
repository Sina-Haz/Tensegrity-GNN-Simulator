from jax import numpy as np, vmap
from quat import *
from data import Robot
import jraph

"""
In this file focus on writing functions which can transform data from r.b. level (i.e. robot)
to the GNN node and edge level
"""

MAX_HEIGHT = 0.5
CONTACT_THRESHOLD = 2e-1

######### DATA TRANSFORMS
@jax.jit
def GlobalToLocal(global_coords, body_positions, body_orientations):
    """
    Takes 3 arrays of equal first dimension

    global_coords: shape = (batch_sz, 3), the absolute position of coordinates
    body_positions: shape = (batch_sz, 3), position of the CoM of body each point is in reference to
    body_orientations: shape = (batch_sz, 4), orientation as a quaternion of body each point is in reference to

    returns: local_coords: shape = (batch_sz, 3)
    """
    rot_matrs = quat_as_matrix(body_orientations)
    rot_matr_invs = np.transpose(rot_matrs, axes = (0, 2, 1))

    offset = global_coords - body_positions

    # Add extra dimension to offsets to allow for matrix multiplication element-wise, then squeeze them back down
    local_coords = np.matmul(rot_matr_invs, offset[..., np.newaxis]).squeeze(-1)
    return local_coords


@jax.jit
def LocalToGlobal(local_coords, body_positions, body_orientations):
    """
    Takes 3 arrays of equal first dimension

    local_coords: shape = (batch_sz, 3), the position of coordinates in body space
    body_positions: shape = (batch_sz, 3), position of the CoM of body each point is in reference
    body_orientations: shape = (batch_sz, 4), orientation as a quaternion of body we want to compute local reference w.r.t.

    returns: global_coords: shape = (batch_sz, 3)
    """
    rot_matrs = quat_as_matrix(body_orientations)

    # Add extra dimension to coordinates so that we can do an element wise matrix multiply to rotate each point, then squeeze back down
    rot_coords = np.matmul(rot_matrs, local_coords[..., np.newaxis]).squeeze(-1)

    global_coords = rot_coords + body_positions
    return global_coords

@jax.jit
def normalize(data, axis=-1, thresh=1e-6):
    norm = np.linalg.norm(data, axis=axis, keepdims=True)
    return np.where(norm > 1e-6, data / norm, np.zeros_like(data))

@jax.jit
def node_P(rod_P, rod_Q, loc_nodes):
    """
    Compute node positions in world frame

    Args:
        rod_P: np.array with shape (n_rods, 3)
        rod_Q: np.array with shape (n_rods, 4)
        loc_nodes: np.array with shape (n_rods, n_rbs, 3)
    
    Returns:
        world_nodes: np.array with shape (n_rods, n_rbs, 3)
    """
    # Define a helper function which takes loc_nodes for 1 rod and p, q for 1 rod and maps to global
    def LocToGlob(loc, p, q):
        # so loc is shape (n_rbs, 3), p and q shape (3,)
        p, q = p[np.newaxis, ...], q[np.newaxis, ...] # need to add a "batch dimension" for these

        # Specify None for p and q s.t. they remain constant and we batch this for n_rbs in loc
        return vmap(LocalToGlobal, in_axes=(0, None, None))(loc, p, q)
    
    wrld_nodes = vmap(LocToGlob, in_axes=(0, 0, 0))(loc_nodes, rod_P, rod_Q).squeeze(axis=2)
    return wrld_nodes


def node_P_flattened(rod_P, rod_Q, loc_nodes):
    """
    Compute node positions in world frame

    Args:
        rod_P: np.array with shape (n_rods, 3)
        rod_Q: np.array with shape (n_rods, 4)
        loc_nodes: np.array with shape (n_rods, n_rbs, 3)
    
    Returns:
        world_nodes: np.array with shape (n_rods, n_rbs, 3)
    """
    n_rods, n_rbs, _ = loc_nodes.shape

    # Flatten local_nodes: from (n_rods, n_rbs, 3) to (n_rods*n_rbs, 3)
    local_nodes_flat = loc_nodes.reshape(-1, 3)
    
    # Repeat rod_P and rod_Q for each node.
    # Each rod's position/orientation is repeated n_rbs times.
    rod_P_flat = np.repeat(rod_P, n_rbs, axis=0)  # shape (n_rods*n_rbs, 3)
    rod_Q_flat = np.repeat(rod_Q, n_rbs, axis=0)  # shape (n_rods*n_rbs, 4)

    # Compute global nodes using LocalToGlobal
    global_nodes_flat = LocalToGlobal(local_nodes_flat, rod_P_flat, rod_Q_flat)
    
    # Reshape back to (n_rods, n_rbs, 3)
    world_nodes = global_nodes_flat.reshape(n_rods, n_rbs, 3)
    return world_nodes


@jax.jit
def get_prev(P, Q, V, W, dt):
    """
    Based on current state, approximate previous state using current velocity (linear & angular)

    Args:
        P: np.array with shape (n_rods, 3)
        Q: np.array with shape (n_rods, 4)
        V: np.array with shape (n_rods, 3)
        W: np.array with shape (n_rods, 3)
        dt: timestep to go backwards
    """
    P_prev = P - V * dt

    # Convert W to quaternion form
    W_as_q = np.zeros_like(Q)
    W_as_q.at[:, 1:].set(-W)

    Q_prev = quat_mul_batch(Q, quat_exp(W_as_q * 0.5 * dt))

    return P_prev, Q_prev


# To apply ctrl: ctrl signal, min/max rest_len, max_w/prev_w/winch_r of motor, curr_rest_len
def applyControl(ctrl, curr_rest_len, prev_ws, m_speed, winch_r, max_w, min_rl, max_rl,  dt):
    """
    Applies a control to the motors controlling the cables for actuation -> this changes the cables' active
    rest length -> then based on the current length of the cable we can calculate the cable force from this actuation

    ctrl: np array of shape (num_cables, ), inputs to motors of actuated cables and 0's for non-actuated in range [-1, 1]
    curr_rest_len: np array of shape (num_cables, ) stores the current rest length of cables
    prev_ws: np array of shape (num_cables, ) stores previous motor speed or 0 for actuated vs non-actuated cables
    m_speed: np array of shape (num_cables, ) -> an input param from [0, 1] to scale ctrl inputs
    winch_r: float representing winch radius of motors
    max_w: float representing maximum angular motor speed
    min_rl: float representing minimum cable rest length
    max_rl: float representing maximum cable rest length
    dt: float representing timestep

    Returns:
        new_rest_len: np array of shape (num_cables, ) of the current rest lengths of both actuated + non-actuated cables
    """
    # Compute new motor speed given control and get the average over the time interval
    new_ws = ctrl * m_speed * max_w 
    avg_ws = (prev_ws + new_ws) / 2.0

    # With this we can get the change in rest length caused by the motor activation and update to get the new rest lengths
    delta_len = avg_ws * winch_r * dt
    new_rest_len = curr_rest_len - delta_len

    # Clamp this within cable limits
    new_rest_len = np.clip(new_rest_len, min=min_rl, max=max_rl)

    return new_rest_len



def node_features(node_P, node_P_prev, rod_P, inv_M, inv_I, endcap_R, local_P, dt):
    """
    Compute the node features for building the input graph of our GNN.
    NOTE: Comparable to _compute_node_feats and _compute_shape_feats in batch_tensegrity_data_processor
    Args:
        node_P: np array with shape (n_rods, n_rbs, 3)
        node_P_prev: np array with shape (n_rods, n_rbs, 3)
        rod_P: np array with shape (n_rods, 3)
        inv_M: np array with shape (n_rods * n_rbs, 1)
        inv_I: np array with shape (n_rods * n_rbs, 3)
        endcap_R: float representing radius of the endcaps
        local_P: np array of shape (n_rods, n_rbs, 3) of nodes in local frame
        dt: float representing timestep len

    Returns:
        PyTree (dict) with features embedded
    """
    # Compute nodes' offset from rods' center by adding a dimension on rod_P and using broadcasting along dim=1
    rod_P_repeated = np.repeat(rod_P, node_P.shape[1], axis=0)
    node_P, node_P_prev,  =node_P.reshape(-1, 3), node_P_prev.reshape(-1, 3),

    # diff = node_P - rod_P[:, None, :]
    diff = node_P - rod_P_repeated
    # Now use this to compute distance along last dimension (x,y,z)
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    # Lastly we compute the unit direction of nodes to their respective rod position
    dir = np.where(dist > 1e-6, diff/dist, np.zeros_like(diff))

    # Compute velocity approx from prev and current node positions
    node_V = (node_P - node_P_prev) / dt

    # Now we calculate the height of each node above the ground (which is at z=0)
    node_z = node_P[:, 2:3] # shape (n_nodes, 1)
    ht_above_grnd = np.minimum(node_z - endcap_R, MAX_HEIGHT)

    # Put all of these features into a PyTree:
    node_feats = {
        "pos": node_P,
        "vel": node_V,
        "height": ht_above_grnd,
        "inv_M": inv_M,
        "inv_I": inv_I,
        "com_dir": dir,
        "com_dist": dist,
        "body_frame": local_P.reshape(-1, 3),
    }


    # For adding the ground node we simply add zeros of correct shape to each of the features
    node_feats = jax.tree.map(lambda leaf: np.concat([leaf, np.zeros_like(leaf[0].reshape(1, -1))], axis=0), node_feats)
    # node_feats['height'][-1] = -endcap_R # to match nelson's engine, may be important
    return node_feats

def frame_features(node_pos, frame_idx1, frame_idx2):
    """
    Helper function which calculates node relative distance and direction in a frame defined by 1st and last nodes,
    is aligned with the robot orientation wise and respects it's rod-sphere based geometry. 

    NOTE: This helper function is akin to compute_shape_feats under batch_tensegrity_data_processor 

    node_pos: np array of shape (n_nodes, 3) doesn't matter if it has ground node or not
    frame_idx1: int for indexing node_pos -> sphere 0 of the first tensegrity rod
    frame_idx2: int for indexing node_pos -> sphere 1 of the last tensegrity rod
    """
    n_nodes = node_pos.shape[0]

    # use the weird indexing s.t. indexing array of shape (x, 3) gives (1, 3) instead of (3,)
    n1 = np.repeat(node_pos[np.array([frame_idx1])], n_nodes, axis=0)
    n2 = np.repeat(node_pos[np.array([frame_idx2])], n_nodes, axis=0)

     # x_dir is just n2 - n1 projected onto XY plane
    x_dir = (n2-n1) * np.array([1., 1., 0.])
    x_dir = normalize(x_dir)

    # z-direction = principal axis (z-axis)
    z_dir = np.repeat(np.array([[0., 0., 1.]], dtype=x_dir.dtype), n_nodes, axis=0)

    # y-direction is cross prod of these two to ensure orthogonality
    y_dir = np.cross(z_dir, x_dir) # order has to be flipped to get different sign

    # Using this create rotation matrices
    R = np.stack([x_dir, y_dir, z_dir], axis=2)

    # Ok now, we compute offset of every node position from n1 and then apply R^T to re-orient into the new frame
    offset = (np.transpose(R, axes=(0, 2, 1)) @ (node_pos - n1)[..., np.newaxis]).squeeze()
    offset_norm = np.linalg.norm(offset, axis=-1, keepdims=True)

    return offset, offset_norm


def filter_contacts(contact_edges, node_pos, endcap_R, threshold):
    """
    Computes which contact nodes (i.e. nodes representing an endcap) are close to the ground

    contact_edges: np array of shape (2, n_cons * 2) b/c undirected edges
    node_pos: np array of shape (n_nodes+1, 3) includes the grnd node at the end
    endcap_R: float representing radius of the endcaps 
    threshold: float representing how close a contact should be to the ground if
    """
    dist_to_grnd = np.abs(node_pos[contact_edges[1], 3] - node_pos[contact_edges[0], 3]) # shape (n_cons*2)
    close = (dist_to_grnd - endcap_R) < threshold # creates boolean array of which contacts are close enough
    filtered = contact_edges[:, close] # idx along 2nd dim for edges which are close to the grnd

    return filtered


def edge_features(node_feats, adj_list, edge_type, curr_rest_len, ks, kd, endcap_R):
    """
    Compute edge feats of different types of edges including body, cable, and contact edges
    Assumes that applyControl has been called and rest len is updated

    node_feats: PyTree representing all the node features (used mainly for edge len computing)
    adj_list: np array of shape (2, n_total_edges). list of all edges b/w nodes, undirected so its times 2
    edge_type: np array of shape (n_total_edges) 0 -> body edge, 1 -> cable edge, 2 -> contact edge
    curr_rest_len: np array of shape (2*n_cables, 1) obtained by np.repeat(applyControl() ,2)
    ks: np aray of shape (2*n_cables, 1) representing stiffness of cables for cable_edges
    kd: np array of shape (2*n_cables, 1) representing damping coeff. of cables for cable_edges
    endcap_R: float representing endcap radius of the rods
    """
    node_pos = node_feats['pos']
    node_vels = node_feats['vel']
    loc_nodes = node_feats['body_frame']

    # Compute body edge features: current displacement/dist. based on node pos + start displacement/dist.
    E_b = adj_list[:, edge_type==0]
    curr_disp = node_pos[E_b[1]] - node_pos[E_b[0]]
    curr_dist = np.linalg.norm(curr_disp, axis=1, keepdims=True)
    rest_disp = loc_nodes[E_b[1]] - loc_nodes[E_b[0]]
    rest_dist = np.linalg.norm(rest_disp, axis=1, keepdims=True)

    # Compute cable edge features: displacement+dist+dir b/w endcaps, relative velocity, rest len, stiffness, and damping
    E_cable = adj_list[:, edge_type==1]
    cable_disp = node_pos[E_cable[0]] - node_pos[E_cable[1]]
    cable_dist = np.linalg.norm(cable_disp, axis=1, keepdims=True)
    cable_dir = cable_disp / (cable_dist+1e-8)
    ks, kd, curr_rest_len = ks[..., np.newaxis], kd[..., np.newaxis], curr_rest_len[..., np.newaxis]
    # Get relative velocity and use elt-wise dot product to see how aligned it is with cable direction 
    cable_relV = node_vels[E_cable[1]] - node_vels[E_cable[0]]
    cable_relV_norm = np.expand_dims(np.einsum('ij, ij -> i', cable_relV, cable_dir), axis=1) # vectorized dotproduct
    stiffness_mag = (cable_dist - curr_rest_len) * ks
    damping_mag = kd * cable_relV_norm

    # Compute contact edge feats: min dist to ground, contact normal vector, normal+tangential comp. of rel. vel.
    E_con = adj_list[:, edge_type == 2]
    # Get edge directionality (tensegrity -> grnd is neg., grnd -> tensegrity is pos.)
    edge_dir, half = np.ones(shape=(E_con.shape[1],1)), E_con.shape[1] // 2
    edge_dir = edge_dir.at[:half].multiply(-1)
    con_dist = (node_pos[E_con[1]][:, 2:3] - node_pos[E_con[0]][:, 2:3]) - edge_dir * endcap_R
    con_relV = node_vels[E_con[1], :] - node_vels[E_con[0], :]
    # Contact normal direction is then based solely on the z-axis:
    con_norm_dir = np.tile(np.array([0,0,1], dtype=node_pos.dtype), (E_con.shape[1], 1)) * edge_dir # contact normal direction (unit vector)
    # Now we can get the normal and tangential component of the relative velocity
    con_norm_relV = np.expand_dims(np.vecdot(con_relV, con_norm_dir, axis=1), axis=1) # normal comp of rel. V computed via dot prod.
    con_tan_dir = con_relV - con_norm_dir * con_norm_relV # remaining is tangential vector
    con_tan_relV = np.linalg.norm(con_tan_dir, axis=1, keepdims=True)+1e-8 # tangential comp. of rel. V based on it's norm
    con_tan_dir = con_tan_dir / con_tan_relV # normalize so tangential direction is unit vector

    body_feats = {
        'body_curr_disp': curr_disp,
        'body_curr_dist': curr_dist,
        'body_rest_disp': rest_disp,
        'body_rest_dist': rest_dist
    }
    cable_feats = {
        'cable_disp': cable_disp,
        'cable_dist': cable_dist,
        'cable_dir': cable_dir,
        'cable_relV': cable_relV,
        'cable_relV_norm': cable_relV_norm,
        'cable_stiffness': ks,
        'cable_damping': kd,
        'cable_stiffness_mag': stiffness_mag,
        'cable_damping_mag': damping_mag,
        'cable_rest_len': curr_rest_len
    }
    con_feats = {
        'con_dist': con_dist,
        'con_normal_dir': con_norm_dir,
        'con_normal_relV': con_norm_relV,
        'con_tan_dir': con_tan_dir,
        'con_tan_relV': con_tan_relV
    }


    edge_feats = {
        'body': body_feats,
        'cable': cable_feats,
        'con': con_feats
    }

    return edge_feats
    





def build_graph(robot: Robot, dt: float):
    """
    Takes robot with rigid body data, transforms to node and edge level

    Graph Representation:
    Nodes: PyTree/feature matrix -> N[i] of PyTree gives all feature vectors for i-th node
    Edges: 
        1. Adjacency list A of shape (2, n_edges) where A[k] = (node_i_idx, node_j_idx)
        2. PyTree/Feature matrix E -> E[i] gives all the feature vectors of i-th edge
    """
    # Compute current node position in world frame, approx. previous node position in world frame
    node_pos = node_P(robot.P, robot.Q, robot.local_nodes)

    p_prev, q_prev = get_prev(robot.P, robot.Q, robot.V, robot.W, dt)
    prev_node_pos = node_P(p_prev, q_prev, robot.local_nodes)

    # Compute node features based on this and other robot information
    node_f = node_features(node_pos, prev_node_pos, robot.P, robot.inv_M, robot.inv_I,
                           robot.endcap_R, robot.local_nodes, dt)
        
    node_pos = node_f['pos'] # use the node pos with ground node included (also flattened)

    offsets, offset_norms = frame_features(node_pos, robot.frame_idx_1, robot.frame_idx_2)
    node_f["local_offsets"] =  offsets
    node_f["local_dists"] = offset_norms

    # Now onto getting our edge indices for graph
    # First we want to filter the contact edges from all possible to get a set shape
    contact_edges = filter_contacts(robot.contact_edges, node_pos, robot.endcap_R, CONTACT_THRESHOLD)

    # Next we concatenate all the edges together along axis 1 + store extra edge type int vector
    # 0 = body edges, 1 = cable edges, 2 = contact edges
    adj_list = np.concat([robot.body_edges, robot.cable_edges, contact_edges], axis=1)
    n_body_edges, n_cable_edges, n_con_edges = robot.body_edges.shape[1], robot.cable_edges.shape[1], contact_edges.shape[1]
    edge_types = np.concat([np.zeros((n_body_edges,), dtype=np.int32), np.ones((n_cable_edges, ), dtype=np.int32), 2*np.ones((n_con_edges, ), dtype=np.int32)], axis=0)

    # Compute edge features:
    edge_f = edge_features(node_f, adj_list, edge_types, robot.rest_len, robot.ks, robot.kd, robot.endcap_R)
    edge_f['edge_type'] = edge_types

    # Graph Representation: Nodes -> node feature matrix/dict, Edges -> Adjacency list + Edge feature dict
    graph = jraph.GraphsTuple(
        nodes=node_f,
        edges=edge_f,
        senders=adj_list[0],
        receivers=adj_list[1],
        n_node=np.array([node_pos.shape[0]]),
        n_edge=np.array([adj_list.shape[1]]),
        globals=None
    )
    return graph












