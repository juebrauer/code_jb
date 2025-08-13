import time
import pybullet as p
import pybullet_data as pd  # for using URDF models

# 1. connect with PyBullet 
cid = p.connect(p.GUI) # p.GUI or p.DIRECT (headless) mode?

# 2. make sure, PyBullet can find the URDF models used in the following
p.setAdditionalSearchPath(pd.getDataPath())
print(f"AdditionalSearchPath = ", pd.getDataPath())

# 3. prepare the simulation
p.resetSimulation()
p.setGravity(0, 0, -9.81) # acceleration in x/y/z coordinates
p.setTimeStep(1.0/240.0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # do not show GUI-Overlays

# 4. add two URDFs
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("r2d2.urdf", [0, 0, 1.5])

# 5. simulation loop
for _ in range(10 * 240):  # simulate for 10 seconds
    p.stepSimulation()
    time.sleep(1.0/240.0)  # let GUI time to update

# 6. close connection to PyBullet
p.disconnect()