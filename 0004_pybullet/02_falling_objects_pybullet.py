# Start:
#    python 02_falling_objects_pybullet.py --n 30 --seconds 15 --seed 42

import argparse
import math
import random
import time
import tkinter as tk

import pybullet as p
import pybullet_data as pd

def screen_size():
    """Get screen size with tkinter (platform independent)."""
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def spawn_random_body():
    """
    Creates randomly a sphere or a box with random geometry, color and material parameters
    """

    # 1. decide randomly whether to create a sphere or a box
    shape_type = random.choice(["sphere", "box"])

    # 2.1 generate a sphere
    if shape_type == "sphere":
        radius = random.uniform(0.05, 0.20)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                  rgbaColor=(random.random(), random.random(), random.random(), 1.0))
        # let mass depend roughly on the volume
        mass = (4.0 / 3.0) * math.pi * radius**3 * 250.0

    # 2.2 generate a box
    elif shape_type == "box":
        hx = random.uniform(0.05, 0.25)
        hy = random.uniform(0.05, 0.25)
        hz = random.uniform(0.05, 0.25)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                  rgbaColor=(random.random(), random.random(), random.random(), 1.0))
        mass = (2 * hx) * (2 * hy) * (2 * hz) * 250.0

    # 3. set random start pose: (x,y,z) position and orientation (yaw,pitch,roll) Euler angles
    x = random.uniform(-2.0, 2.0)
    y = random.uniform(-2.0, 2.0)
    z = random.uniform(2.5, 7.0)
    yaw = random.uniform(-math.pi, math.pi)
    pitch = random.uniform(-0.5, 0.5)
    roll = random.uniform(-0.5, 0.5)
    orn = p.getQuaternionFromEuler([roll, pitch, yaw]) # Euler angles to quaternion

    # 4. create the body
    body = p.createMultiBody(baseMass=mass,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[x, y, z],
                             baseOrientation=orn)

    # 5. change material properties, in order to see interesting collision behaviors
    p.changeDynamics(body, -1,
                     lateralFriction=random.uniform(0.3, 1.0),
                     rollingFriction=random.uniform(0.0, 0.02),
                     spinningFriction=random.uniform(0.0, 0.02),
                     restitution=random.uniform(0.1, 0.9))
    return body


def main():

    # 1. get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="nr of objects to fall down")
    parser.add_argument("--seconds", type=float, default=10.0, help="time to simulate")
    parser.add_argument("--seed", type=int, default=None, help="random seed value")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 2. connect with GUI
    maximize_window = True
    if maximize_window:    
        win_w, win_h = screen_size()
        cid = p.connect(p.GUI, options=f"--width={int(win_w)} --height={int(win_h)} --window_position=0,0")
    else:
        cid = p.connect(p.GUI)    
    assert cid >= 0, "PyBullet GUI could not be started!"

    # 3. set additional search path for URDF model access
    p.setAdditionalSearchPath(pd.getDataPath())

    # 4. prepare scene
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.setPhysicsEngineParameter(numSolverIterations=150,
                                enableConeFriction=1,
                                contactSlop=0.001)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # do not show GUI-Overlays

    # 5. load floor
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=0.95, restitution=0.4)

    # 6. generate n objects
    bodies = [spawn_random_body() for _ in range(args.n)]

    # 7. add info in GUI
    p.addUserDebugText(f"{args.n} objects – Press ESC to exit",
                       textPosition=[0, 0, 2.0], textSize=1.5, lifeTime=2.5)

    # 8. simulation loop
    steps = int(args.seconds * 240)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # 9. clean-up
    p.disconnect()


if __name__ == "__main__":
    main()
