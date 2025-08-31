# Overview

Kinematic chain simulator and reinforcement learning in order to learn how to a reach an end-effector position.



# `01_kinematic_chain.py`

Just a kinematic chain simulation with N DOFs and N arms.

Example usage:\
`python 01_kinematic_chain.py 5`



# `02_kinematic_chain_rl.py`

Kinematic chain simulation and a Deep RL agent learns to set the angles of the DOFs of the kinematic chain such that a goal end-effector position is reached: it learns Inverse Kinematics (IK)

Example usage:\
`python 02_kinematic_chain_rl.py 1`
