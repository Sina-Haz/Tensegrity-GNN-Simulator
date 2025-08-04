import sys
import os
import json

# Get the absolute path of the codebase root (one level above `gnn_physics`)
codebase_root = os.path.abspath("tensegrity_gnn_compiled")

# Add it to sys.path
sys.path.insert(0, codebase_root)

# Now you can safely import
import torch
from torch_geometric.data import Data as GraphData
import jax.numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")

# Import your BatchTensegrityDataProcessor and TensegrityRobot
# (Make sure these modules are in your PYTHONPATH)
from gnn_physics.data_processors.batch_tensegrity_data_processor import BatchTensegrityDataProcessor
from state_objects.cables import ActuatedCable
from robots.tensegrity import TensegrityRobotGNN
from data import *
from transforms import *

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

def convert_graph(graph: GraphData):
    """
    Take in Nelson's graph and convert it to a jraph GraphsTuple so that it can be compared
    """
    

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
    # print(graph)

    graph2 = build_graph(robot, 0.1)
    print(graph2)


if __name__ == "__main__":
    main()