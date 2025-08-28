# Overview

This is a dataset that can be used to train a robot arm with machine learning.

Filename: `data_franka_pick_and_place_100episodes.zip`
File size: 3.5 GB
Download: [Download link](https://datasetsml.sfo3.cdn.digitaloceanspaces.com/data_franka_pick_and_place_100episodes.zip)

It contains 100 episodes in which a [Franka Robotics](https://franka.de/) robot arm stands on a table with some blocks. One of these blocks has a red color and the robot tries to pick it (using inverse kinematics) and place it then in a green small region on the table.


# Directory structure

The dataset has the following directory structure:

<pre>
data_franka_pick_and_place_100episodes
    |
    ├── episode_0000
    │   ├── actions.csv (What did the robot do in each time step?)
    │   ├── initial_positions.csv (Where are all the blocks on the table at start time?)
    │   ├── validation_result.csv (Did the robot succeed in placing the red block onto the green area?)
    │   ├── corner (Camera images from corner view)
    │   │   ├── corner_000000.png
    │   │   ├── corner_000010.png
    │   │   ├── corner_000020.png
    │   │   ├── ...   
    │   ├── front (Camera images from front view)
    │   │   ├── front_000000.png
    │   │   ├── front_000010.png
    │   │   ├── front_000020.png
    │   │   ├── ...
    │   ├── side (Camera images from side view)
    │   │   ├── side_000000.png
    │   │   ├── side_000010.png
    │   │   ├── side_000020.png
    │   │   ├── ...
    │   ├── top (Camera images from top view)
    │   │   ├── top_000000.png
    │   │   ├── top_000010.png
    │   │   ├── top_000020.png
    │   │   ├── ...
    |
    ├── episode_0001
    |   |
    ... ...
</pre>

# Action file: `actions.csv`

<pre><code>
frame,top_image,side_image,front_image,corner_image,ee_tx,ee_ty,ee_tz,ee_qx,ee_qy,ee_qz,ee_qw,q0,q1,q2,q3,q4,q5,q6,gripper_open_width
0,top/top_000000.png,side/side_000000.png,front/front_000000.png,corner/corner_000000.png,-0.04298173263669014,-7.566724207208608e-07,1.211266279220581,1.0,0.0001981087843887508,-1.2660880202020053e-05,3.7853360481676646e-06,0.0,-0.785,0.0,-2.356,0.0,1.571,0.785,0.08
10,top/top_000010.png,side/side_000010.png,front/front_000010.png,corner/corner_000010.png,-0.04298173263669014,-7.566724207208608e-07,1.211266279220581,1.0,0.0001981087843887508,-1.2660880202020053e-05,3.7853360481676646e-06,0.0,-0.785,0.0,-2.356,0.0,1.571,0.785,0.08
20,top/top_000020.png,side/side_000020.png,front/front_000020.png,corner/corner_000020.png,-0.04298173263669014,-7.566724207208608e-07,1.211266279220581,1.0,0.0001981087843887508,-1.2660880202020053e-05,3.7853360481676646e-06,0.0,-0.785,0.0,-2.356,0.0,1.571,0.785,0.08
30,top/top_000030.png,side/side_000030.png,front/front_000030.png,corner/corner_000030.png,-0.04298173263669014,-7.566724207208608e-07,1.211266279220581,1.0,0.0001981087843887508,-1.2660880202020053e-05,3.7853360481676646e-06,0.0,-0.785,0.0,-2.356,0.0,1.571,0.785,0.08
...
1960,top/top_001960.png,side/side_001960.png,front/front_001960.png,corner/corner_001960.png,-0.04999999999999999,0.0,0.87,1.0,0.0,0.0,6.123233995736766e-17,0.0431575873023303,-0.2681420264154849,-0.10233458813429759,-2.9141646856756926,-0.045604519181810604,2.6449366470758746,0.7784066692224725,0.08
1970,top/top_001970.png,side/side_001970.png,front/front_001970.png,corner/corner_001970.png,-0.04999999999999999,0.0,0.87,1.0,0.0,0.0,6.123233995736766e-17,0.05434487896014903,-0.30979437828915,-0.0973472741958661,-2.9649091777565246,-0.05650237562264207,2.654789856251295,0.8018064414971104,0.08
1980,top/top_001980.png,side/side_001980.png,front/front_001980.png,corner/corner_001980.png,-0.04999999999999999,0.0,0.87,1.0,0.0,0.0,6.123233995736766e-17,0.06553217061796782,-0.35144673016281525,-0.09235996025743459,-3.015653669837357,-0.06740023206347354,2.6646430654267155,0.8252062137717484,0.08
</code></pre>

| Column name | Meaning |
|------|-----------|
| `frame` | Sequential frame number/timestep in the episode |
| `corner_image` | Relative file path to PNG image from corner/bird's eye camera view |
| `front_image` | Relative file path to PNG image from front camera view |
| `side_image` | Relative file path to PNG image from side camera view |
| `top_image` | Relative file path to PNG image from top-down camera view |
| `ee_tx` | End-effector x-position in 3D space (meters) |
| `ee_ty` | End-effector y-position in 3D space (meters) |
| `ee_tz` | End-effector z-position in 3D space (meters) |
| `ee_qx` | End-effector orientation quaternion x-component |
| `ee_qy` | End-effector orientation quaternion y-component |
| `ee_qz` | End-effector orientation quaternion z-component |
| `ee_qw` | End-effector orientation quaternion w-component |
| `q0` | Joint 0 angle (radians) - Base rotation |
| `q1` | Joint 1 angle (radians) - Shoulder pitch |
| `q2` | Joint 2 angle (radians) - Shoulder roll |
| `q3` | Joint 3 angle (radians) - Elbow pitch |
| `q4` | Joint 4 angle (radians) - Wrist roll |
| `q5` | Joint 5 angle (radians) - Wrist pitch |
| `q6` | Joint 6 angle (radians) - Wrist roll |
| `gripper_open_width` | Opening width of gripper fingers (meters, 0.0 = closed, ~0.08 = fully open) |


# Initial block positions file: `initial_positions.csv`

The x,y start coordinates are random. There is exactly one red block and 10 gray blocks.

<pre><code>
object_id,body_id,color,x,y,z
0,4,red,0.01662806352645485,-0.2630949711427573,0.643
1,5,gray,-0.22182082664259412,0.034289273672044374,0.643
2,6,gray,0.1778715919042465,0.22383410060401276,0.643
3,7,gray,-0.08529199906310692,-0.12963618421080378,0.643
4,8,gray,-0.14571177278509995,0.13554639124019163,0.643
5,9,gray,0.2218458891713705,-0.07855272174488001,0.643
6,10,gray,-0.2646407131545718,0.13822123158000743,0.643
7,11,gray,-0.22913800135415163,-0.12903367583536682,0.643
8,12,gray,-0.32177438365738176,0.19254412326698755,0.643
9,13,gray,0.28649613881402747,0.14167222218555958,0.643
10,14,gray,0.15923145963935548,0.02925745576069727,0.643
11,15,gray,0.034274053172241636,0.07607357263480563,0.643
</code></pre>


# Validation result file: `validation_result.csv`

<pre><code>
Metric,Value
success,True
red_position,"(0.2434, -0.2343, 0.6430)"
green_center,"(0.2500, -0.2500)"
green_size,"(0.1800, 0.1400)"
x_in_bounds,True
y_in_bounds,True
z_correct,True
x_offset_from_center,-0.0066
y_offset_from_center,0.0157
z_offset_from_expected,-0.0000
</code></pre>