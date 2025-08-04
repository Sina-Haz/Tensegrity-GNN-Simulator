import sys
import os
import json
"""
This is a test file which may help show how to use the code base to generate graphs from states etc
"""
# Get the absolute path of the codebase root (one level above `gnn_physics`)
codebase_root = os.path.abspath("tensegrity_gnn_compiled")

# Add it to sys.path
sys.path.insert(0, codebase_root)

# Now you can safely import
import torch
from torch_geometric.data import Data as GraphData
import jax.numpy as np
import numpy as n
import jax
import pandas as pd
jax.config.update("jax_platform_name", "cpu")

# Import your BatchTensegrityDataProcessor and TensegrityRobot
# (Make sure these modules are in your PYTHONPATH)
from gnn_physics.data_processors.batch_tensegrity_data_processor import BatchTensegrityDataProcessor
from state_objects.cables import ActuatedCable
from robots.tensegrity import TensegrityRobotGNN
from data import *
from transforms import *
from save import *

def get_curr_state(ten: TensegrityRobotGNN) -> torch.Tensor:
    return torch.hstack([
        torch.hstack([
            rod.pos,
            rod.quat,
            rod.linear_vel,
            rod.ang_vel,
        ])
        for rod in ten.rods.values()
    ])

def compare_graphs(graph1: GraphData, graph2: jraph.GraphsTuple):
    """
    Take in Nelson's graph and convert it to a jraph GraphsTuple so that it can be compared

    Notes:
    node_dir_from_com: Not equal b/c this is a unit vector where the magnitudes we divide by can be very small
        thus causing numerical instability even though the node positions are exactly the same and 
        the error on distances is very small

    node_dist_from_com: Again this is error is small, on the order of 1e-6/1e-7, likely caused by difference
        in precision handling and certain math implementations b/w jax and torch

    node_dist_to_ground: 
        This is b/c in nelson's code the distance to grnd for grnd node is -sphere_radius (or -endcap_R)

    

    """
    # graph.edge_index[0] = senders, graph.edge_index[1] = recievers
    # Maps are nelson -> mine, I give what shape nelsons are
    node_map = {
        'pos': 'pos', # shape (n_nodes, 3) vs. (n_nodes, 3)
        # 'num_nodes': 'n_nodes', # int vs int
        'vel': 'vel', # shape (n_nodes, 3) vs. (n_nodes, 3)
        'node_inv_mass': 'inv_M', # shape (n_nodes, 1) vs. (n_nodes, 1)
        'node_inv_inertia': 'inv_I', # shape (n_nodes, 3) vs. (n_nodes, 3)
        'node_dir_from_com' : 'com_dir', # shape (n_nodes, 3) vs. (n_nodes, 3)
        'node_dist_from_com_norm': 'com_dist', # shape (n_nodes, 1) vs. (n_nodes, 1)
        'node_dist_to_ground': 'height', # shape (n_nodes, 1) vs. (n_nodes, 1)
        'node_body_verts': 'body_frame', # shape (n_nodes, 3) vs. (n_nodes, 3)
        'node_dist_to_first_node': 'local_offsets', # shape (n_nodes, 3) vs. (n_nodes, 3)
        'node_dist_to_first_node_norm': 'local_dists', # shape (n_nodes, 1) vs. (n_nodes, 1)
    }

    edge_map = {
        'edge_type': 'edge_type', # shape (n_edges, 1) vs (n_edges, 1)
        'body_dist': 'body_curr_disp', # shape (n_body_edges, 3) vs. (n_body_edges, 3)
        'body_dist_norm': 'body_curr_dist', # shape (n_body_edges, 1) vs (n_body_edges, 1)
        'body_rest_dist': 'body_rest_disp', # shape (n_body_edges, 3) vs. (n_body_edges, 3)
        'body_rest_dist_norm': 'body_rest_dist', # shape (n_body_edges, 1) vs. (n_body_edges, 1)
        'contact_dist': 'con_dist', # (n_con_edges, 1) vs. (n_con_edges, 1) 
        'contact_normal': 'con_normal_dir', # (n_con_edges, 3) vs. (n_con_edges, 3) 
        'contact_tangent': 'con_tan_dir', # (n_con_edges, 3) vs. (n_con_edges, 3) 
        'contact_rel_vel_normal': 'con_normal_relV', # (n_con_edges, 1) vs. (n_con_edges, 1) 
        'contact_rel_vel_tangent': 'con_tan_relV', # (n_con_edges, 1) vs. (n_con_edges, 1) 
        'cable_dist': 'cable_disp', # (n_cable_edges, 3) vs. (n_cable_edges, 3)
        'cable_dist_norm': 'cable_dist', # (n_cable_edges, 1) vs. (n_cable_edges, 1) 
        'cable_dir': 'cable_dir', # (n_cable_edges, 3) vs. (n_cable_edges, 3)
        'cable_rel_vel_norm': 'cable_relV_norm', # (n_cable_edges, 1) vs. (n_cable_edges, 1)
        'cable_rest_length': 'cable_rest_len', # (n_cable_edges, 1) vs. (n_cable_edges, 1)
        'cable_stiffness': 'cable_stiffness', # (n_cable_edges, 1) vs. (n_cable_edges, 1)
        'cable_damping': 'cable_damping', # (n_cable_edges, 1) vs. (n_cable_edges, 1)
        'cable_stiffness_force_mag': 'cable_stiffness_mag', # (n_cable_edges, 1) vs. (n_cable_edges, 1)
        'cable_damping_force_mag': 'cable_damping_mag' # (n_cable_edges, 1) vs. (n_cable_edges, 1)
       }
    
    for key, value in node_map.items():
        arr1 = graph1[key].numpy() if isinstance(graph1[key], torch.Tensor) else np.array([graph1[key]], dtype=np.float32)
        arr2 = graph2.nodes[value]
        if (arr1.shape == arr2.shape):
            close = np.allclose(arr1, arr2)
            print(f'Results for {key}: {close}')
            if not close:
                print(np.max(np.abs(arr2 - arr1)))
        else:
            print(f'Results for {key}: shapes dont match with {value}')

    print('\n------------\n')
    for key, value in edge_map.items():
        arr1 = graph1[key].numpy() if isinstance(graph1[key], torch.Tensor) else np.array([graph1[key]], dtype=np.float32)
        arr2 = graph2.edges[value]
        if (arr1.shape == arr2.shape):
            close = np.allclose(arr1, arr2)
            print(f'Results for {key}: {close}')
            if not close:
                print(np.max(np.abs(arr2 - arr1)))
                print(f'Nelsons {arr1}, Mine {arr2}')
        else:
            print(f'Results for {key}: shapes dont match with {value}, shapes: {arr1.shape, arr2.shape}')



def main():

    # Create a TensegrityRobot instance using tensegrity cfg found in codebase
    with open('tensegrity_gnn_compiled/simulators/configs/3_bar_tensegrity_gnn_sim_config.json', 'r') as file:
        cfg = json.load(file)

    cfg = cfg['tensegrity_cfg']
    ten = TensegrityRobotGNN(cfg)
    dp = BatchTensegrityDataProcessor(ten)

    # Our own data class for robot
    robot = Robot.TensegrityToRobot(ten)


    
 
    # Compare graphs -> ensure values across features are the same, use the current state
    # Absolute error should be < 1e-6, compare it on the tensor level, node feats, edge feats, adj list

    graph1 = dp.batch_state_to_graph([get_curr_state(ten)])
    # # Iterate over all attributes
    # for key, value in graph1:
    #     print(f"{key}: {value.shape if isinstance (value, torch.Tensor) else value}")
    #     print('________')

    graph2 = build_graph(robot, 0.1)
    # # Iterate over node features
    # if isinstance(graph2.nodes, dict):
    #     for feat_name, value in graph2.nodes.items():
    #         print(f"{feat_name}: {value.shape if isinstance(value, np.ndarray) else value}")

    # # Iterate over edge features
    # if isinstance(graph2.edges, dict):
    #     for feat_name, value in graph2.edges.items():
    #         print(f"{feat_name}: {value.shape if isinstance(value, np.ndarray) else value}")
    
    save_graph_tuple(graph2, filename='tst_graph_2.bin')
    graph3 = load_graph_tuple('tst_graph.bin')
    # print(graphs_equal(graph2, graph3))
    
if __name__ == "__main__":
    main()