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
4. 