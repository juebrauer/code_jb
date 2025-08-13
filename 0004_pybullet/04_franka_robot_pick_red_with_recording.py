import os
import math, time, random, csv
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data as pd

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
DT = 1.0 / 240.0
ARM_FORCE = 120.0
FINGER_FORCE = 40.0

TABLE_TOP_Z = 0.62                 # table height (top surface)
TABLE_HALF = (0.60, 0.40, 0.04)    # half-dimensions of tabletop (x, y, z)
OBJ_HALF = 0.02                    # 2 cm half-edge length
OBJ_MASS = 0.05                    # 50 g
N_OBJECTS = 12

# Colors
GRAY = (0.5, 0.5, 0.5, 1.0)
RED  = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 0.8, 0.0, 1.0)       # Farbe für die Zielfläche

# Spacing / safety margins
EDGE_MARGIN = 0.12                 # keep objects away from table edges (meters)
CLEARANCE_RED = 0.06               # min distance any gray object must keep from the red object
CLEARANCE_ANY = 0.04               # min mutual distance between any two objects
RED_MIN_DIST_FROM_ROBOT = 0.28     # keep red object away from robot base

# Data capture
IMG_W, IMG_H = 640, 480
DATA_ROOT = "data"                  # folder to store episodes

# If True, attach a fixed constraint on grasp to guarantee pick
STRONG_GRASP_HACK = False

# Global variable to store camera state between episodes
CAMERA_STATE = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a value x into the closed interval [lo, hi]."""
    return max(lo, min(hi, x))


def step(n: int = 1):
    """Advance the physics simulation by n time steps with real-time sleep."""
    for _ in range(n):
        p.stepSimulation()
        time.sleep(DT)


def set_friction(body: int, link: int = -1, lateral: float = 1.0,
                 spinning: float = 0.001, rolling: float = 0.001) -> None:
    """Set contact friction parameters for a given body/link."""
    p.changeDynamics(body, link, lateralFriction=lateral,
                     spinningFriction=spinning, rollingFriction=rolling)


def create_table() -> int:
    """Create a simple kinematic tabletop as a box and return the body id."""
    hx, hy, hz = TABLE_HALF
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                              rgbaColor=[0.75, 0.55, 0.35, 1])
    body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[0, 0, TABLE_TOP_Z - hz])
    set_friction(body, -1, lateral=1.0)
    return body


def create_green_target_surface(center=(0.25, -0.25), size=(0.16, 0.12)) -> Tuple[int, Tuple[float, float, float]]:
    """Create a green flat surface on the table where the red block should be placed.
    
    Args:
        center: (x, y) center of the green surface on the table.
        size: (x, y) dimensions of the green surface.
    
    Returns:
        (surface_body_id, drop_position) where drop_position is the target position
        to place objects on the surface.
    """
    cx, cy = center
    sx, sy = size
    thickness = 0.003  # Sehr dünne Fläche
    
    # Erstelle die grüne Zielfläche
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx * 0.5, sy * 0.5, thickness * 0.5])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx * 0.5, sy * 0.5, thickness * 0.5], 
                              rgbaColor=GREEN)
    surface = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=[cx, cy, TABLE_TOP_Z + thickness * 0.5])
    
    # Die Drop-Position ist leicht über der grünen Fläche
    drop_pos = (cx, cy, TABLE_TOP_Z + thickness + OBJ_HALF + 0.001)
    
    return surface, drop_pos


def spawn_boxes_gray_and_one_red(n: int, seed: Optional[int] = None,
                                 avoid_center: Optional[Tuple[float, float, float]] = None,
                                 avoid_radius: float = 0.0,
                                 clearance_red: float = CLEARANCE_RED,
                                 clearance_any: float = CLEARANCE_ANY,
                                 green_surface_center: Tuple[float, float] = None,
                                 green_surface_size: Tuple[float, float] = None) -> Tuple[List[int], int]:
    """Spawn n small cubes on the tabletop, all gray except one red, with spacing rules.
    
    Args:
        n: number of cubes to spawn.
        seed: optional random seed for reproducibility.
        avoid_center: a 3D point to keep the red object away from (e.g., robot base).
        avoid_radius: minimum distance the red object must keep from avoid_center.
        clearance_red: minimal distance any gray cube must keep from the red cube.
        clearance_any: minimal distance between any two cubes.
        green_surface_center: center of the green target surface to avoid.
        green_surface_size: size of the green target surface.
    
    Returns:
        (list_of_body_ids, red_body_id)
    """
    if seed is not None:
        random.seed(seed)

    def rand_xy():
        hx, hy, _ = TABLE_HALF
        XMIN, XMAX = -hx + EDGE_MARGIN, hx - EDGE_MARGIN
        YMIN, YMAX = -hy + EDGE_MARGIN, hy - EDGE_MARGIN
        return random.uniform(XMIN, XMAX), random.uniform(YMIN, YMAX)
    
    def is_on_green_surface(x, y):
        """Check if position (x, y) is on the green surface."""
        if green_surface_center is None or green_surface_size is None:
            return False
        cx, cy = green_surface_center
        sx, sy = green_surface_size
        return (abs(x - cx) < sx * 0.5 + OBJ_HALF and 
                abs(y - cy) < sy * 0.5 + OBJ_HALF)

    # Sample red position first with constraints
    max_tries = 2000
    for _ in range(max_tries):
        rx, ry = rand_xy()
        rz = TABLE_TOP_Z + OBJ_HALF + 0.003
        
        # Avoid robot base
        if avoid_center is not None:
            dx = rx - avoid_center[0]
            dy = ry - avoid_center[1]
            dz = (TABLE_TOP_Z - avoid_center[2]) if len(avoid_center) == 3 else 0.0
            if math.sqrt(dx*dx + dy*dy + dz*dz) < avoid_radius:
                continue
        
        # Avoid green surface
        if is_on_green_surface(rx, ry):
            continue
            
        red_pos = (rx, ry, rz)
        break
    else:
        raise RuntimeError("Could not sample a valid red object position.")

    positions = [red_pos]

    # Now sample the remaining gray objects with mutual clearances
    while len(positions) < n:
        for _ in range(max_tries):
            x, y = rand_xy()
            z = TABLE_TOP_Z + OBJ_HALF + 0.003
            
            # Avoid green surface
            if is_on_green_surface(x, y):
                continue
                
            pos = (x, y, z)
            ok = True
            for j, q in enumerate(positions):
                dx = pos[0] - q[0]
                dy = pos[1] - q[1]
                d = math.hypot(dx, dy)
                req = clearance_red if j == 0 else clearance_any  # first is red
                if d < req:
                    ok = False
                    break
            if ok:
                positions.append(pos)
                break
        else:
            # Relax spacing a bit if crowded
            clearance_any = max(0.5 * clearance_any, 0.02)

    # Create bodies
    bodies = []
    for i, pos in enumerate(positions):
        color = RED if i == 0 else GRAY
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[OBJ_HALF] * 3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[OBJ_HALF] * 3, rgbaColor=color)
        b = p.createMultiBody(baseMass=OBJ_MASS, baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis, basePosition=list(pos))
        set_friction(b, -1, lateral=1.2)
        bodies.append(b)

    red_body_id = bodies[0]
    return bodies, red_body_id


def get_link_index_by_name(body: int, needle: str) -> Optional[int]:
    """Return the first link index whose name contains the given substring."""
    for j in range(p.getNumJoints(body)):
        name = p.getJointInfo(body, j)[12].decode()
        if needle in name:
            return j
    return None


def joint_indices(body: int, include_fingers: bool = False):
    """Return joint indices for revolute/prismatic joints and optionally the finger joints."""
    idxs = []
    fingers = []
    for j in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, j)
        jtype = info[2]
        name = info[1].decode()
        if "finger" in name:
            fingers.append(j)
            continue
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            idxs.append(j)
    if include_fingers:
        return idxs, fingers
    return idxs


def save_camera_state():
    """Save the current camera state (GUI only)."""
    global CAMERA_STATE
    try:
        debug_info = p.getDebugVisualizerCamera()
        CAMERA_STATE = {
            'distance': debug_info[10],
            'yaw': debug_info[8],
            'pitch': debug_info[9],
            'target': debug_info[11]
        }
    except:
        pass  # Ignore errors in headless mode


def restore_camera_state():
    """Restore the previously saved camera state (GUI only)."""
    global CAMERA_STATE
    if CAMERA_STATE is not None:
        try:
            p.resetDebugVisualizerCamera(
                cameraDistance=CAMERA_STATE['distance'],
                cameraYaw=CAMERA_STATE['yaw'],
                cameraPitch=CAMERA_STATE['pitch'],
                cameraTargetPosition=CAMERA_STATE['target']
            )
        except:
            pass  # Ignore errors in headless mode


class DataRecorder:
    """Utility to record camera frames and action vectors for imitation learning.

    For each simulation step, call `next_frame(action_vector)` to save images from
    top and side cameras and append the action vector into a CSV file.
    """

    def __init__(self, root: str, ep_idx: int, top_view, top_proj, side_view, side_proj, img_w: int = IMG_W, img_h: int = IMG_H, save_every: int = 1):
        self.ep_idx = ep_idx
        self.img_w = img_w
        self.img_h = img_h
        self.top_view = top_view
        self.top_proj = top_proj
        self.side_view = side_view
        self.side_proj = side_proj
        self.frame = 0
        self.save_every = max(1, int(save_every))

        # Create directories
        self.base_dir = os.path.join(root, f"episode_{ep_idx:04d}")
        self.top_dir = os.path.join(self.base_dir, "top")
        self.side_dir = os.path.join(self.base_dir, "side")
        os.makedirs(self.top_dir, exist_ok=True)
        os.makedirs(self.side_dir, exist_ok=True)

        # Open CSV for action logging
        self.csv_path = os.path.join(self.base_dir, "actions.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        header = [
            "frame",            # global sim frame index (not the saved-frame index)
            "top_image",
            "side_image",
            # Action vector fields:
            "ee_tx", "ee_ty", "ee_tz",
            "ee_qx", "ee_qy", "ee_qz", "ee_qw",
            # Joint targets (7 for Panda)
            "q0","q1","q2","q3","q4","q5","q6",
            # Gripper open width command (meters)
            "gripper_open_width",
        ]
        self.writer.writerow(header)

    def _save_camera(self, view, proj, path: str) -> None:
        """Capture current scene with provided camera matrices and save PNG to path.
        NOTE: pybullet.getCameraImage returns (w, h, rgbImg, depthImg, segImg).
        We must read the *third* return value for the RGBA buffer.
        """
        w, h, rgba, _, _ = p.getCameraImage(self.img_w, self.img_h, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgba = np.array(rgba, dtype=np.uint8)
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]
        Image.fromarray(rgb).save(path)

    def next_frame(self, action_vec: List[float]) -> None:
        """Capture both cameras and append the given action vector for the current frame.
        Respects `save_every`: only every k-th global frame is saved.
        """
        should_save = (self.frame % self.save_every == 0)
        if should_save:
            top_name = f"top_{self.frame:06d}.png"
            side_name = f"side_{self.frame:06d}.png"
            top_path = os.path.join(self.top_dir, top_name)
            side_path = os.path.join(self.side_dir, side_name)
            self._save_camera(self.top_view, self.top_proj, top_path)
            self._save_camera(self.side_view, self.side_proj, side_path)
            row = [self.frame, os.path.relpath(top_path, self.base_dir), os.path.relpath(side_path, self.base_dir)] + action_vec
            self.writer.writerow(row)
        self.frame += 1

    def close(self):
        """Close any file handles held by the recorder."""
        try:
            self.csv_file.close()
        except Exception:
            pass


def set_gripper(robot: int, finger_joints: List[int], open_width: float = 0.08, speed_steps: int = 60,
                recorder: Optional[DataRecorder] = None, arm_joint_targets: Optional[List[float]] = None,
                ee_target: Optional[Tuple[List[float], List[float]]] = None) -> None:
    """Open/close the gripper fingers to a target opening width.

    Args:
        robot: robot body id.
        finger_joints: indices of finger joints.
        open_width: target opening width (meters). Each finger moves half of it.
        speed_steps: number of small control steps to reach the target.
        recorder: optional DataRecorder to log (images, actions) each step.
        arm_joint_targets: optional arm joint targets to include in the action vector.
        ee_target: optional (pos, orn) end-effector target to include in the action vector.
    """
    tgt = clamp(open_width * 0.5, 0.0, 0.045)
    for _ in range(speed_steps):
        p.setJointMotorControlArray(robot, finger_joints, p.POSITION_CONTROL,
                                    targetPositions=[tgt] * len(finger_joints),
                                    forces=[FINGER_FORCE] * len(finger_joints))
        step(1)
        if recorder is not None:
            # Build an action vector to log (fallbacks if not provided)
            if arm_joint_targets is None:
                # Use current arm positions
                arm_idxs = joint_indices(robot)
                arm_joint_targets_now = [p.getJointState(robot, j)[0] for j in arm_idxs]
            else:
                arm_joint_targets_now = arm_joint_targets

            if ee_target is None:
                # Use current hand pose
                ee = get_link_index_by_name(robot, "panda_hand")
                ls = p.getLinkState(robot, ee, computeForwardKinematics=True)
                ee_pos, ee_orn = ls[4], ls[5]
            else:
                ee_pos, ee_orn = ee_target

            action_vec = list(ee_pos) + list(ee_orn) + list(arm_joint_targets_now[:7]) + [open_width]
            recorder.next_frame(action_vec)


def move_ik(robot: int, arm_joints: List[int], ee_link: int, pos: List[float], orn: List[float], duration_s: float = 1.0,
            recorder: Optional[DataRecorder] = None, gripper_open_width: float = 0.08) -> List[float]:
    """Smoothly move the arm using IK to reach the target pose (pos, orn).

    Args:
        robot: robot body id.
        arm_joints: indices of the 7 DOF arm joints.
        ee_link: end-effector link index.
        pos: target position [x, y, z].
        orn: target orientation quaternion [x, y, z, w].
        duration_s: motion duration in seconds.
        recorder: optional DataRecorder to log (images, actions) per step.
        gripper_open_width: last commanded gripper opening, included in the action vector.

    Returns:
        The final joint target array used (length = number of arm joints).
    """
    steps = max(1, int(duration_s / DT))
    q_start = [p.getJointState(robot, j)[0] for j in arm_joints]
    
    # Use more robust IK with current joint positions as rest pose
    # This helps to find consistent solutions across episodes
    q_goal = p.calculateInverseKinematics(
        robot, ee_link, pos, orn,
        restPoses=q_start,  # Use current position as rest pose
        maxNumIterations=100,
        residualThreshold=1e-5
    )
    q_goal = list(q_goal[:len(arm_joints)])

    for k in range(steps):
        a = (k + 1) / steps
        q = [s + a * (t - s) for s, t in zip(q_start, q_goal)]
        p.setJointMotorControlArray(robot, arm_joints, p.POSITION_CONTROL,
                                    targetPositions=q, forces=[ARM_FORCE] * len(arm_joints))
        step(1)
        if recorder is not None:
            action_vec = list(pos) + list(orn) + list(q[:7]) + [gripper_open_width]
            recorder.next_frame(action_vec)

    return q_goal


def setup_cameras() -> Tuple[List[float], List[float], List[float], List[float]]:
    """Create view/projection matrices for a top and a side camera.

    Returns:
        (top_view, top_proj, side_view, side_proj)
    """
    # Top camera: above the table, looking down
    cam_target = [0.05, 0.0, TABLE_TOP_Z]
    cam_up = [1, 0, 0]  # x-axis up to avoid roll ambiguity
    top_eye = [0.05, 0.0, TABLE_TOP_Z + 0.75]
    top_view = p.computeViewMatrix(cameraEyePosition=top_eye,
                                   cameraTargetPosition=cam_target,
                                   cameraUpVector=cam_up)
    top_proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=IMG_W / IMG_H, nearVal=0.01, farVal=3.0)

    # Side camera: from +y side, looking towards the table center
    side_eye = [-0.1, 0.75, TABLE_TOP_Z + 0.25]
    side_target = [0.1, 0.0, TABLE_TOP_Z + 0.05]
    side_up = [0, 0, 1]
    side_view = p.computeViewMatrix(cameraEyePosition=side_eye,
                                    cameraTargetPosition=side_target,
                                    cameraUpVector=side_up)
    side_proj = p.computeProjectionMatrixFOV(fov=70.0, aspect=IMG_W / IMG_H, nearVal=0.01, farVal=3.0)

    return top_view, top_proj, side_view, side_proj


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def run_episode(ep_idx: int, gui: bool = True, seed: Optional[int] = None, save_every: int = 1, first_episode: bool = False) -> None:
    """Run a single episode: spawn objects, pick the red one, place on green surface, and record data.

    Args:
        ep_idx: episode index (used for seeding and file naming).
        gui: whether to use PyBullet GUI.
        seed: optional random seed for object placement.
        save_every: save only every k-th frame (k>=1) to disk.
        first_episode: True if this is the first episode in a sequence.
    """
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Restore camera state if not first episode
    if not first_episode and gui:
        restore_camera_state()

    # Environment
    p.loadURDF("plane.urdf")
    _table = create_table()
    panda_base_pos = [-0.35, 0.0, TABLE_TOP_Z + 0.001]
    panda = p.loadURDF("franka_panda/panda.urdf", basePosition=panda_base_pos, useFixedBase=True)

    # Increase finger friction
    left_finger = get_link_index_by_name(panda, "leftfinger")
    right_finger = get_link_index_by_name(panda, "rightfinger")
    for lf in [left_finger, right_finger]:
        if lf is not None:
            set_friction(panda, lf, lateral=2.0)

    # Joints and EE
    arm = joint_indices(panda)
    _, fingers = joint_indices(panda, include_fingers=True)
    ee = get_link_index_by_name(panda, "panda_hand")
    assert ee is not None, "End-effector link not found"

    # Green target surface (replacing container)
    green_center = (0.25, -0.25)
    green_size = (0.18, 0.14)
    _, drop_pos = create_green_target_surface(center=green_center, size=green_size)

    # Objects with spacing constraints (avoiding green surface)
    objs, red_obj = spawn_boxes_gray_and_one_red(
        N_OBJECTS, seed=seed,
        avoid_center=tuple(panda_base_pos),
        avoid_radius=RED_MIN_DIST_FROM_ROBOT,
        clearance_red=CLEARANCE_RED,
        clearance_any=CLEARANCE_ANY,
        green_surface_center=green_center,
        green_surface_size=green_size
    )

    # Wait for settling
    step(240)

    # Home pose to a comfortable middle configuration
    mid = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Standard home position for Panda
    p.setJointMotorControlArray(panda, arm[:7], p.POSITION_CONTROL, targetPositions=mid, forces=[ARM_FORCE] * 7)
    step(int(1.0 / DT))  # Give more time to reach home position

    # Cameras & recorder
    top_view, top_proj, side_view, side_proj = setup_cameras()
    recorder = DataRecorder(DATA_ROOT, ep_idx, top_view, top_proj, side_view, side_proj, IMG_W, IMG_H, save_every=save_every)

    # Open gripper (record action)
    set_gripper(panda, fingers, open_width=0.08, speed_steps=80, recorder=recorder,
                arm_joint_targets=mid, ee_target=(p.getLinkState(panda, ee)[4], p.getLinkState(panda, ee)[5]))

    # Determine target pose above red object
    pos_obj, _ = p.getBasePositionAndOrientation(red_obj)
    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])  # TCP points down
    approach = [pos_obj[0], pos_obj[1], TABLE_TOP_Z + 0.18]
    descend  = [pos_obj[0], pos_obj[1], TABLE_TOP_Z + OBJ_HALF + 0.005]  # Get closer to object
    lift     = [pos_obj[0], pos_obj[1], TABLE_TOP_Z + 0.25]

    # Approach, descend, close, lift
    q_last = move_ik(panda, arm, ee, approach, down_orn, duration_s=1.2, recorder=recorder, gripper_open_width=0.08)
    q_last = move_ik(panda, arm, ee, descend,  down_orn, duration_s=1.0, recorder=recorder, gripper_open_width=0.08)
    
    # Small pause to stabilize before grasping
    for _ in range(int(0.3 / DT)):
        step(1)
        if recorder is not None:
            action_vec = list(descend) + list(down_orn) + list(q_last[:7]) + [0.08]
            recorder.next_frame(action_vec)

    # Close gripper and optionally constrain
    set_gripper(panda, fingers, open_width=0.0, speed_steps=120, recorder=recorder,
                arm_joint_targets=q_last, ee_target=(descend, down_orn))

    grasp_cid = None
    if STRONG_GRASP_HACK:
        hand_state = p.getLinkState(panda, ee, computeForwardKinematics=True)
        hand_pos, hand_orn = hand_state[4], hand_state[5]
        grasp_cid = p.createConstraint(parentBodyUniqueId=panda, parentJointIndex=ee,
                                       childBodyUniqueId=red_obj, childLinkIndex=-1,
                                       jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                       parentFramePosition=[0, 0, 0],
                                       childFramePosition=p.getBasePositionAndOrientation(red_obj)[0])
    step(int(0.2 / DT))

    # Lift
    q_last = move_ik(panda, arm, ee, lift, down_orn, duration_s=1.0, recorder=recorder, gripper_open_width=0.0)

    # Move to green surface
    above_drop = [drop_pos[0], drop_pos[1], lift[2]]
    q_last = move_ik(panda, arm, ee, above_drop, down_orn, duration_s=1.0, recorder=recorder, gripper_open_width=0.0)
    
    # Lower to place on green surface (adjusted height for surface instead of container)
    lower_drop = [drop_pos[0], drop_pos[1], drop_pos[2] + 0.005]  # Slightly above the calculated drop position
    q_last = move_ik(panda, arm, ee, lower_drop, down_orn, duration_s=0.8, recorder=recorder, gripper_open_width=0.0)

    # Open to release
    set_gripper(panda, fingers, open_width=0.08, speed_steps=80, recorder=recorder,
                arm_joint_targets=q_last, ee_target=(lower_drop, down_orn))
    if grasp_cid is not None:
        p.removeConstraint(grasp_cid)

    step(int(0.6 / DT))

    # Save camera state before closing
    if gui:
        save_camera_state()

    # Cleanup
    recorder.close()
    time.sleep(0.2)
    p.disconnect()


def main(gui: bool = True, episodes: int = 3, seed: Optional[int] = 42, save_every: int = 1) -> None:
    """Main entry: run multiple episodes of pick-and-place data collection.

    Args:
        gui: if True, run with PyBullet GUI; if False, headless.
        episodes: number of episodes to record.
        seed: base seed; each episode uses seed + ep_idx for variability.
        save_every: save only every k-th frame to disk (k>=1).
    """
    os.makedirs(DATA_ROOT, exist_ok=True)
    for ep in range(episodes):
        run_seed = None if seed is None else (seed + ep)
        run_episode(ep, gui=gui, seed=run_seed, save_every=save_every, first_episode=(ep == 0))


if __name__ == "__main__":
    # Set gui=False for headless data collection
    main(gui=True, episodes=2, seed=123, save_every=50)