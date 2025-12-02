import math, time, random
import numpy as np
import pybullet as p
import pybullet_data as pd

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
DT = 1.0/240.0              # Physics simulation time step [s]. PyBullet's GUI defaults to 240 Hz.
                            # With realTimeSimulation=0 we advance time explicitly by p.stepSimulation(),
                            # so each step advances the world by DT seconds.

ARM_FORCE = 120.0           # Max motor force (torque/force limit depending on joint type) for arm joints.
FINGER_FORCE = 40.0         # Max motor force for the finger joints.

TABLE_TOP_Z = 0.62          # Height of the table top in world Z [m].
TABLE_HALF = (0.60, 0.40, 0.04)  # Half extents of the table block [m]: (x/2, y/2, z/2).
                                # Full size would be (1.20 x 0.80 x 0.08) m.

N_OBJECTS = 12               # How many cubes to scatter.
OBJ_HALF = 0.02              # Half edge length of a cube [m] => cube is 4 cm on a side.
OBJ_MASS = 0.05              # Cube mass [kg] (50 g). Gravity will apply ~0.05 * 9.81 ≈ 0.49 N downward.

# RGBA colors for the cubes. Alpha=1 means fully opaque.
COLORS = [
    (1,0,0,1),(0,1,0,1),(0,0,1,1),(1,1,0,1),(1,0,1,1),(0,1,1,1),
    (1,0.5,0,1),(0.6,0.2,1,1),(0.9,0.2,0.2,1),(0.2,0.8,0.4,1)
]

STRONG_GRASP_HACK = False    # If True, we add a fixed constraint when "grasping" to guarantee the hold,
                             # bypassing physics-based friction. Useful for demos if friction is insufficient.

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def clamp(x, lo, hi):
    """Clamp value x into the range [lo, hi]."""
    return max(lo, min(hi, x))

def step(n=1):
    """
    Advance the physics by n * DT seconds.
    We also sleep for DT per step so wall-clock time roughly matches simulation time in GUI mode.
    """
    for _ in range(n):
        p.stepSimulation()
        time.sleep(DT)

def set_friction(body, link=-1, lateral=1.0, spinning=0.001, rolling=0.001):
    """
    Tune contact/friction properties of a body (or specific link).
    - lateralFriction: Coulomb friction coefficient μ for tangential (sliding) resistance.
    - spinningFriction: resists in-place spinning (about contact normal).
    - rollingFriction: resists rolling without slip.
    """
    p.changeDynamics(body, link, lateralFriction=lateral,
                     spinningFriction=spinning, rollingFriction=rolling)

def create_table():
    """
    Create a simple rectangular table top as a *static* rigid body.
    We use half extents to define the collision/visual box. The base position we pass is the COM.
    For a centered box, placing COM at z = TABLE_TOP_Z - hz puts the top surface at TABLE_TOP_Z.
    """
    hx, hy, hz = TABLE_HALF
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                              rgbaColor=[0.75, 0.55, 0.35, 1])
    # Static body (baseMass=0) so gravity doesn’t move it and it doesn’t need motors.
    body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[0, 0, TABLE_TOP_Z - hz])
    set_friction(body, -1, lateral=1.0)
    return body

def spawn_colored_boxes(n=N_OBJECTS, seed=0):
    """
    Spawn n small cubes on the table with random XY positions.
    Z is set slightly above the table so they "settle" by gravity without interpenetration.
    """
    random.seed(seed)
    bodies = []
    hx, hy, hz = TABLE_HALF
    # Define a spawn rectangle inset from edges to avoid spawning partially off the table.
    XMIN, XMAX = -hx + 0.10, hx - 0.10
    YMIN, YMAX = -hy + 0.10, hy - 0.10
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[OBJ_HALF]*3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[OBJ_HALF]*3, rgbaColor=color)
        x = random.uniform(XMIN, XMAX)
        y = random.uniform(YMIN, YMAX)
        # Place the cube's COM so the bottom face is ~2 mm above the table:
        # bottom_z = TABLE_TOP_Z + 0.002 -> COM_z = bottom_z + OBJ_HALF
        z = TABLE_TOP_Z + OBJ_HALF + 0.002
        b = p.createMultiBody(baseMass=OBJ_MASS, baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis, basePosition=[x, y, z])
        set_friction(b, -1, lateral=1.2)  # Slightly higher μ to make grasping easier.
        bodies.append(b)
    return bodies

def get_link_index_by_name(body, needle):
    """
    Utility to find a link index by a (partial) name match.
    Joint info tuple: name bytes at index 12. We decode and compare substrings.
    """
    for j in range(p.getNumJoints(body)):
        name = p.getJointInfo(body, j)[12].decode()
        if needle in name:
            return j
    return None

def joint_indices(body, include_fingers=False):
    """
    Collect indices of revolute/prismatic joints (i.e., actuated configuration DOFs).
    Optionally separate out 'finger' joints by name.
    """
    idxs = []
    fingers = []
    for j in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, j)
        jtype = info[2]               # p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, etc.
        name = info[1].decode()
        if "finger" in name:
            fingers.append(j); continue
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            idxs.append(j)
    if include_fingers:
        return idxs, fingers
    return idxs

def set_gripper(robot, finger_joints, open_width=0.08, speed_steps=60):
    """
    Open/close the gripper by commanding symmetric finger positions.
    The user-facing 'open_width' is the total jaw width; each finger moves half of it.
    We clamp to 0..0.045 because Panda fingers have a limited stroke (~45 mm/side).
    We send the same target repeatedly over 'speed_steps' to animate the motion smoothly.
    """
    tgt = clamp(open_width * 0.5, 0.0, 0.045)  # per-finger target [m]
    for _ in range(speed_steps):
        # POSITION_CONTROL drives each joint toward targetPositions with given force limit.
        p.setJointMotorControlArray(robot, finger_joints, p.POSITION_CONTROL,
                                    targetPositions=[tgt]*len(finger_joints),
                                    forces=[FINGER_FORCE]*len(finger_joints))
        step(1)  # advance physics by DT

def move_ik(robot, arm_joints, ee_link, pos, orn, duration_s=1.0):
    """
    Smoothly move the arm from current joint configuration to an IK solution for (pos,orn).
    - pos: desired end-effector position [x,y,z] in world frame [m].
    - orn: desired end-effector orientation as a quaternion [x,y,z,w].
    IK returns a joint vector q_goal; we then *linearly interpolate in joint space*
    from q_start to q_goal over 'steps' frames to create a time-parameterized trajectory.

    Note: Linear in joint space != linear in task space, but is simple and robust for demos.
    """
    steps = max(1, int(duration_s/DT))  # number of discrete control frames
    # Read current joint positions (q_start).
    q_start = [p.getJointState(robot, j)[0] for j in arm_joints]
    # Compute IK. PyBullet's calculateInverseKinematics solves for a feasible joint vector.
    q_goal = p.calculateInverseKinematics(robot, ee_link, pos, orn)
    q_goal = q_goal[:len(arm_joints)]   # trim in case IK returns more joints than we use
    for k in range(steps):
        # 'a' goes from (1/steps) to 1. This is a simple ramp for interpolation.
        a = (k+1)/steps
        # Joint-space interpolation: q(k) = (1-a)*q_start + a*q_goal
        q = [s + a*(t-s) for s, t in zip(q_start, q_goal)]
        p.setJointMotorControlArray(robot, arm_joints, p.POSITION_CONTROL,
                                    targetPositions=q, forces=[ARM_FORCE]*len(arm_joints))
        step(1)  # advance one control frame

# ------------------------------------------------------------
# Main simulation logic
# ------------------------------------------------------------
def main(gui=True):
    # Connect to the physics server (GUI or DIRECT headless).
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())  # so default URDFs can be found
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)   # Standard Earth gravity [m/s^2].
    p.setTimeStep(DT)           # Fixed-step integrator with our DT.
    p.setRealTimeSimulation(0)  # Manual stepping (deterministic with step()).
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Hide built-in overlays for a cleaner view.

    # Ground plane + table
    p.loadURDF("plane.urdf")
    table = create_table()

    # Place the Franka Panda on the table.
    # We add +0.001 m in Z to avoid z-fighting (coplanar rendering artifacts).
    panda_base_pos = [-0.35, 0.0, TABLE_TOP_Z + 0.001]
    panda = p.loadURDF("franka_panda/panda.urdf", basePosition=panda_base_pos,
                       useFixedBase=True)  # Fixed base so the arm doesn't topple.

    # Increase fingertip friction to improve grip (higher μ reduces slip under tangential load).
    left_finger = get_link_index_by_name(panda, "leftfinger")
    right_finger = get_link_index_by_name(panda, "rightfinger")
    for lf in [left_finger, right_finger]:
        if lf is not None:
            set_friction(panda, lf, lateral=2.0)

    # Get joint indices: arm (7 DOF) and finger joints.
    arm = joint_indices(panda)          # revolute/prismatic joints excluding fingers
    _, fingers = joint_indices(panda, include_fingers=True)

    # Find the end-effector link (the "panda_hand" frame).
    ee = get_link_index_by_name(panda, "panda_hand")
    assert ee is not None, "End-effector link not found"

    # Spawn colored objects and let them settle under gravity + contact.
    objs = spawn_colored_boxes(N_OBJECTS, seed=42)
    step(240)  # 240 * DT = 1.0 s of simulated settling time.

    # Move the arm to a "home" pose: midpoint of each joint's limits.
    # If the URDF reports invalid limits (lo>=hi), fall back to [-pi, pi].
    mid = []
    for j in arm:
        lo, hi = p.getJointInfo(panda, j)[8], p.getJointInfo(panda, j)[9]  # joint limits [rad or m]
        if lo >= hi: lo, hi = -math.pi, math.pi
        mid.append(0.5*(lo+hi))  # midpoint = (lo + hi)/2
    p.setJointMotorControlArray(panda, arm, p.POSITION_CONTROL,
                                targetPositions=mid, forces=[ARM_FORCE]*len(arm))
    step(int(0.7/DT))  # ~0.7 s to reach the mid configuration.

    # Open gripper to a wide jaw width (~80 mm total).
    set_gripper(panda, fingers, open_width=0.08, speed_steps=80)

    # Select the first spawned cube as the pick target.
    target = objs[0]
    pos_obj, _ = p.getBasePositionAndOrientation(target)  # (x,y,z) of object's COM and its orientation

    # Define task-space waypoints for a top-down pick:
    # Orientation: end-effector (TCP) pointing down -> 180° rotation about X gives Z-axis flipped.
    # We convert Euler (XYZ order) to a quaternion because PyBullet expects quaternions for orientation.
    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    # Choose heights:
    # - approach: high above the object to avoid collisions on the way in
    # - descend: just above the object's top face = table_top + cube_half + small clearance
    # - lift: raise the object after grasping
    approach = [pos_obj[0], pos_obj[1], TABLE_TOP_Z + 0.18]
    descend  = [pos_obj[0], pos_obj[1], TABLE_TOP_Z + OBJ_HALF + 0.01]
    lift     = [pos_obj[0], pos_obj[1], TABLE_TOP_Z + 0.25]

    # Execute the approach and descend using IK-driven joint-space interpolation.
    move_ik(panda, arm, ee, approach, down_orn, duration_s=1.0)
    move_ik(panda, arm, ee, descend,  down_orn, duration_s=0.8)

    # Close the gripper to grasp.
    # If friction & contact are sufficient, the object should be held by normal + friction forces.
    set_gripper(panda, fingers, open_width=0.0, speed_steps=120)

    grasp_cid = None
    if STRONG_GRASP_HACK:
        """
        Optional "hard attach":
        We create a FIXED joint (constraint) between the hand link and the object.
        This pins the object rigidly to the hand frame and ignores slip physics.
        """
        hand_state = p.getLinkState(panda, ee, computeForwardKinematics=True)
        hand_pos, hand_orn = hand_state[4], hand_state[5]  # world pose of the hand link
        grasp_cid = p.createConstraint(
            parentBodyUniqueId=panda, parentJointIndex=ee,
            childBodyUniqueId=target, childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0,0,0],
            parentFramePosition=[0,0,0],
            # childFramePosition picks the object's current COM position so we don't "snap" it.
            childFramePosition=p.getBasePositionAndOrientation(target)[0]
        )

    step(int(0.2/DT))  # Brief pause so contact constraints converge with the closed fingers.

    # Lift straight up to clear the table; gravity will load the grasp with ~0.49 N downward.
    move_ik(panda, arm, ee, lift, down_orn, duration_s=1.0)
    step(int(0.5/DT))

    # Translate laterally in XY while keeping Z constant (simple re-locate maneuver).
    side = [lift[0] + 0.10, lift[1] + 0.10, lift[2]]
    move_ik(panda, arm, ee, side, down_orn, duration_s=1.0)
    step(int(0.5/DT))

    # Place the object back down:
    # Move down until the cube's bottom is just above the table, then open fingers.
    move_ik(panda, arm, ee, [side[0], side[1], TABLE_TOP_Z + OBJ_HALF + 0.01], down_orn, 0.8)
    set_gripper(panda, fingers, open_width=0.08, speed_steps=80)

    # If we used the hard constraint, remove it now so physics takes over again.
    if grasp_cid is not None:
        p.removeConstraint(grasp_cid)
    step(int(0.6/DT))

    # Small pause (purely cosmetic) and disconnect from physics server.
    time.sleep(10)
    p.disconnect()

if __name__ == "__main__":
    # gui=True opens the interactive viewer; gui=False runs headless (useful for CI).
    main(gui=True)
