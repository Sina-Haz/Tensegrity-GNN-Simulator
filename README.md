### The Project

The goal of this project is to implement a learned physics simulator to simulate the Tensegrity robot and it's complex dynamics. 
For more information on the methods and principles behind the simulator refer to the following [paper](https://arxiv.org/abs/2410.12216)

The research behind the architecture and design of the GNN is not my own, my purpose for this project is to use [JAX](https://github.com/jax-ml/jax) and [Jraph](https://github.com/google-deepmind/jraph)
to build a faster GNN and enable faster graph and data processing to accelerate the simulator and allow it to be more practical for other types of robotics research such as Reinforcement Learning as well as MPC or other closed-loop based methods for robot control.

### Usage

To use the simulator you need an input config file which describes the physical properties of the tensegrity robot. This simulator currently only supports ground contact
and does not support additional obstacles.

A good example of how to use the simulator is in `test.py`, but here are the basic steps:
1. Provide a config file and read it as a JSON (needs to be in a certain format, refer to test.py and the config file it uses)
2. This config file is used to instantiate an instance of the `TensegrityRobotGNN` class from here you can generate `Robot` instance using the classmethod `TensegrityToRobot()`
3. Once you have a `Robot` instance you can use it as input to the `build_graph()` method found in `transforms.py` -> this creates the input to the GNN
4. If you haven't done so already create an instance of the GNN using the graph (refer to `gnn.py`) 
5. Use the GNN to predict the change in velocity of the nodes (nodes are just various points on the robot which have been organized into a graph)
6. Use the change in velocity to update the node velocity and node position based on your timestep (i.e. something like v += dv * dt, p += v * dt)
7. Use a function to go from "node-space" back to "rod space" (i.e. we would map local velocity and position to world frame, and aggregate nodes' positions and velocities to compute rod positions and velocities) -> This is not yet implemented
8. Can apply a control to shorten/loosen the rest lengths of the cables of the robot and change the velocity of the motors they attach to thus actuating it (use `applyControl` in `transforms.py`)
9. Feed the new rest lengths, motor angular velocities, rod positions and velocities to `updateState()` in `data.py` to get an updated Robot instance
10. You can repeat steps 3-9 in a loop to use the GNN to simulate the Tensegrity robot