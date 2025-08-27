import os
import math, time, random, csv
from typing import List, Tuple, Optional, Dict
import argparse

import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data as pd

# CNN control imports
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. CNN control mode disabled.")

# ------------------------------------------------------------
# Parameters (same as original)
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

# CNN control parameters
CNN_CONTROL_STEPS = 200            # Maximum number of control steps in CNN mode
CNN_ACTION_DELAY = 0.05            # Delay between CNN predictions (seconds)

# If True, attach a fixed constraint on grasp to guarantee pick
STRONG_GRASP_HACK = False

# Global variable to store camera state between episodes
CAMERA_STATE = None

# Replay parameters
REPLAY_STEP_TIME = 0.05  # Time to wait after setting each action in replay mode (seconds)


# ------------------------------------------------------------
# CNN Model Definition (copied from 05_train_and_test_cnn.py)
# ------------------------------------------------------------
if TORCH_AVAILABLE:
    class RobotCNN(nn.Module):
        """CNN for robot action prediction from 4 camera views"""
        
        def __init__(self, action_dim=15, image_size=224):
            super(RobotCNN, self).__init__()
            self.image_size = image_size
            self.action_dim = action_dim
            
            # CNN for the combined 4 images (12 channels for 4 RGB images)
            self.features = nn.Sequential(
                # First layer
                nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Second layer
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Third layer
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Fourth layer
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Adaptive pooling for uniform output size
                nn.AdaptiveAvgPool2d((4, 4))
            )
            
            # Fully Connected Layers
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.action_dim)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


# ------------------------------------------------------------
# Helpers (same as original)
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
    """Create a green flat surface on the table where the red block should be placed."""
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
                                 green_surface_size: Tuple[float, float] = None) -> Tuple[List[int], int, List[Dict]]:
    """Spawn n small cubes on the tabletop, all gray except one red, with spacing rules."""
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

    # Create bodies and store initial info
    bodies = []
    initial_positions_info = []
    
    for i, pos in enumerate(positions):
        color = RED if i == 0 else GRAY
        color_name = "red" if i == 0 else "gray"
        
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[OBJ_HALF] * 3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[OBJ_HALF] * 3, rgbaColor=color)
        b = p.createMultiBody(baseMass=OBJ_MASS, baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis, basePosition=list(pos))
        set_friction(b, -1, lateral=1.2)
        bodies.append(b)
        
        # Store initial position info
        initial_positions_info.append({
            'object_id': i,
            'body_id': b,
            'color': color_name,
            'x': pos[0],
            'y': pos[1],
            'z': pos[2]
        })

    red_body_id = bodies[0]
    return bodies, red_body_id, initial_positions_info


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


def setup_cameras() -> dict:
    """Create view/projection matrices for four cameras."""
    cameras = {}
    
    # 1. Top camera: above the table, looking down
    cam_target = [0.05, 0.0, TABLE_TOP_Z]
    cam_up = [1, 0, 0]  # x-axis up to avoid roll ambiguity
    top_eye = [0.05, 0.0, TABLE_TOP_Z + 0.75]
    top_view = p.computeViewMatrix(cameraEyePosition=top_eye,
                                   cameraTargetPosition=cam_target,
                                   cameraUpVector=cam_up)
    top_proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=IMG_W / IMG_H, nearVal=0.01, farVal=3.0)
    cameras['top'] = (top_view, top_proj)

    # 2. Side camera: CORRECTED - now truly parallel to table
    side_eye = [0.0, 0.75, TABLE_TOP_Z + 0.10]
    side_target = [0.0, 0.0, TABLE_TOP_Z + 0.10]
    side_up = [0, 0, 1]
    side_view = p.computeViewMatrix(cameraEyePosition=side_eye,
                                    cameraTargetPosition=side_target,
                                    cameraUpVector=side_up)
    side_proj = p.computeProjectionMatrixFOV(fov=70.0, aspect=IMG_W / IMG_H, nearVal=0.01, farVal=3.0)
    cameras['side'] = (side_view, side_proj)

    # 3. Front camera: viewing robot from the front
    front_eye = [1.0, 0.0, TABLE_TOP_Z + 0.25]
    front_target = [-0.2, 0.0, TABLE_TOP_Z + 0.10]
    front_up = [0, 0, 1]
    front_view = p.computeViewMatrix(cameraEyePosition=front_eye,
                                     cameraTargetPosition=front_target,
                                     cameraUpVector=front_up)
    front_proj = p.computeProjectionMatrixFOV(fov=50.0, aspect=IMG_W / IMG_H, nearVal=0.01, farVal=3.0)
    cameras['front'] = (front_view, front_proj)

    # 4. Corner/bird's eye view: diagonal elevated perspective
    corner_eye = [0.8, 0.8, TABLE_TOP_Z + 0.85]
    corner_target = [-0.05, 0.0, TABLE_TOP_Z + 0.05]
    corner_up = [0, 0, 1]
    corner_view = p.computeViewMatrix(cameraEyePosition=corner_eye,
                                      cameraTargetPosition=corner_target,
                                      cameraUpVector=corner_up)
    corner_proj = p.computeProjectionMatrixFOV(fov=45.0, aspect=IMG_W / IMG_H, nearVal=0.01, farVal=3.0)
    cameras['corner'] = (corner_view, corner_proj)

    return cameras


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


def validate_red_on_green(red_obj: int, green_center: Tuple[float, float], 
                          green_size: Tuple[float, float], tolerance: float = 0.01) -> Tuple[bool, Dict]:
    """Validate if the red object is properly placed on the green surface."""
    # Get final position of red object
    pos, _ = p.getBasePositionAndOrientation(red_obj)
    
    # Check if x,y position is within green surface bounds (with tolerance)
    cx, cy = green_center
    sx, sy = green_size
    
    x_min = cx - sx/2 - tolerance
    x_max = cx + sx/2 + tolerance
    y_min = cy - sy/2 - tolerance
    y_max = cy + sy/2 + tolerance
    
    x_ok = x_min <= pos[0] <= x_max
    y_ok = y_min <= pos[1] <= y_max
    
    # Check if z position is reasonable (object should be resting on or near the surface)
    expected_z = TABLE_TOP_Z + OBJ_HALF + 0.003  # Expected z when resting on surface
    z_ok = abs(pos[2] - expected_z) < 0.02  # Allow 2cm tolerance in z
    
    success = x_ok and y_ok and z_ok
    
    validation_info = {
        'success': success,
        'red_position': pos,
        'green_center': green_center,
        'green_size': green_size,
        'x_in_bounds': x_ok,
        'y_in_bounds': y_ok,
        'z_correct': z_ok,
        'x_offset_from_center': pos[0] - cx,
        'y_offset_from_center': pos[1] - cy,
        'z_offset_from_expected': pos[2] - expected_z
    }
    
    return success, validation_info


# ------------------------------------------------------------
# CNN Control Functions
# ------------------------------------------------------------
if TORCH_AVAILABLE:
    def load_cnn_model(model_path: str, device: torch.device) -> RobotCNN:
        """Load a trained CNN model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading CNN model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        model = RobotCNN(
            action_dim=checkpoint['action_dim'],
            image_size=checkpoint['image_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - Action dim: {checkpoint['action_dim']}")
        print(f"  - Image size: {checkpoint['image_size']}")
        print(f"  - Training loss: {checkpoint['train_loss']:.6f}")
        print(f"  - Validation loss: {checkpoint['val_loss']:.6f}")
        
        return model, checkpoint

    def capture_images_for_cnn(cameras_dict: dict, image_size: int) -> torch.Tensor:
        """Capture images from all 4 cameras and prepare them for CNN input."""
        images = []
        
        # Standard image preprocessing (same as training)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for view in ['corner', 'front', 'side', 'top']:
            if view not in cameras_dict:
                raise ValueError(f"Camera view '{view}' not found in cameras_dict")
            
            view_matrix, proj_matrix = cameras_dict[view]
            
            # Capture image
            w, h, rgba, _, _ = p.getCameraImage(IMG_W, IMG_H, view_matrix, proj_matrix, 
                                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
            
            # Convert to PIL Image
            rgba = np.array(rgba, dtype=np.uint8)
            rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]
            pil_img = Image.fromarray(rgb)
            pil_img = pil_img.resize((image_size, image_size))
            
            # Apply transforms
            img_tensor = transform(pil_img)
            images.append(img_tensor)
        
        # Concatenate all 4 images into one tensor (12 channels)
        combined_image = torch.cat(images, dim=0)  # Shape: [12, H, W]
        combined_image = combined_image.unsqueeze(0)  # Add batch dimension: [1, 12, H, W]
        
        return combined_image

    def apply_cnn_action(robot: int, arm_joints: List[int], finger_joints: List[int], 
                        action_vector: np.ndarray, smoothing_factor: float = 0.3) -> None:
        """Apply the CNN-predicted action vector to the robot."""
        # Parse action vector (15 dimensions)
        if len(action_vector) != 15:
            print(f"Warning: Expected 15-dim action vector, got {len(action_vector)}")
            return
        
        # Extract components
        ee_pos = action_vector[:3]
        ee_quat = action_vector[3:7]  # [x, y, z, w]
        joint_targets = action_vector[7:14]  # 7 joint angles
        gripper_width = action_vector[14]
        
        # Clamp values to reasonable ranges
        ee_pos = np.clip(ee_pos, [-1.0, -1.0, 0.4], [1.0, 1.0, 1.2])
        ee_quat = ee_quat / np.linalg.norm(ee_quat)  # Normalize quaternion
        joint_targets = np.clip(joint_targets, -2.8, 2.8)  # Typical Panda joint limits
        gripper_width = np.clip(gripper_width, 0.0, 0.08)
        
        # Apply joint targets with force control
        p.setJointMotorControlArray(robot, arm_joints[:7], p.POSITION_CONTROL,
                                   targetPositions=joint_targets,
                                   forces=[ARM_FORCE] * 7,
                                   positionGains=[0.8] * 7,  # Slightly lower gains for smoother movement
                                   velocityGains=[0.3] * 7)
        
        # Apply gripper control
        gripper_target = clamp(gripper_width * 0.5, 0.0, 0.045)
        p.setJointMotorControlArray(robot, finger_joints, p.POSITION_CONTROL,
                                   targetPositions=[gripper_target] * len(finger_joints),
                                   forces=[FINGER_FORCE] * len(finger_joints))
        
        return {
            'ee_pos': ee_pos,
            'ee_quat': ee_quat,
            'joint_targets': joint_targets,
            'gripper_width': gripper_width
        }


# ------------------------------------------------------------
# Main Episode Functions
# ------------------------------------------------------------
def run_episode_cnn_control(ep_idx: int, model_path: str, gui: bool = True, 
                           seed: Optional[int] = None, save_video: bool = False,
                           max_steps: int = CNN_CONTROL_STEPS) -> None:
    """Run an episode using CNN control: spawn objects and let the CNN control the robot.
    
    Args:
        ep_idx: episode index for naming and seeding.
        model_path: path to the trained CNN model checkpoint.
        gui: whether to use PyBullet GUI.
        seed: optional random seed for object placement.
        save_video: whether to save video frames of the CNN control.
        max_steps: maximum number of CNN control steps.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. CNN control mode requires PyTorch.")
    
    print(f"\n--- CNN Control Episode {ep_idx} ---")
    print(f"Model: {model_path}")
    print(f"Seed: {seed}")
    print(f"Max steps: {max_steps}")
    
    # Setup PyBullet
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Restore camera state if available
    if gui:
        restore_camera_state()

    # Environment setup
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

    # Green target surface
    green_center = (0.25, -0.25)
    green_size = (0.18, 0.14)
    _, drop_pos = create_green_target_surface(center=green_center, size=green_size)

    # Spawn objects with spacing constraints (avoiding green surface)
    objs, red_obj, initial_positions = spawn_boxes_gray_and_one_red(
        N_OBJECTS, seed=seed,
        avoid_center=tuple(panda_base_pos),
        avoid_radius=RED_MIN_DIST_FROM_ROBOT,
        clearance_red=CLEARANCE_RED,
        clearance_any=CLEARANCE_ANY,
        green_surface_center=green_center,
        green_surface_size=green_size
    )
    
    print(f"Spawned {len(objs)} objects (red object ID: {red_obj})")

    # Wait for physics settling
    step(240)

    # Initialize robot to home position
    home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    p.setJointMotorControlArray(panda, arm[:7], p.POSITION_CONTROL, 
                               targetPositions=home_joints, forces=[ARM_FORCE] * 7)
    step(int(1.0 / DT))

    # Load CNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model, checkpoint = load_cnn_model(model_path, device)
        image_size = checkpoint['image_size']
    except Exception as e:
        print(f"Failed to load model: {e}")
        p.disconnect()
        return

    # Setup cameras
    cameras_dict = setup_cameras()
    
    # Setup video recording if requested
    video_recorder = None
    if save_video:
        from datarecorder_stub import DataRecorder  # You might need to implement this
        video_recorder = DataRecorder(DATA_ROOT, ep_idx, cameras_dict, IMG_W, IMG_H, save_every=1)

    # CNN Control Loop
    print("Starting CNN control...")
    step_count = 0
    action_history = []
    
    with torch.no_grad():
        for step_idx in range(max_steps):
            try:
                # Capture current scene
                combined_image = capture_images_for_cnn(cameras_dict, image_size)
                combined_image = combined_image.to(device)
                
                # Get CNN prediction
                predicted_action = model(combined_image)
                action_vector = predicted_action[0].cpu().numpy()  # Remove batch dimension
                
                # Apply smoothing to avoid jittery movements
                if step_idx > 0 and action_history:
                    prev_action = action_history[-1]
                    smoothing = 0.7  # Higher = more smoothing
                    action_vector = smoothing * action_vector + (1 - smoothing) * prev_action
                
                action_history.append(action_vector.copy())
                
                # Apply action to robot
                action_info = apply_cnn_action(panda, arm, fingers, action_vector)
                
                # Debug output every 20 steps
                if step_idx % 20 == 0:
                    print(f"Step {step_idx:3d}: Action applied")
                    print(f"  EE pos: [{action_info['ee_pos'][0]:.3f}, {action_info['ee_pos'][1]:.3f}, {action_info['ee_pos'][2]:.3f}]")
                    print(f"  Gripper: {action_info['gripper_width']:.3f}")
                
                # Record video frame if requested
                if video_recorder is not None:
                    video_recorder.next_frame(action_vector.tolist())
                
                # Advance simulation
                step(max(1, int(CNN_ACTION_DELAY / DT)))
                step_count += 1
                
                # Check if task might be completed (red object near green surface)
                red_pos, _ = p.getBasePositionAndOrientation(red_obj)
                distance_to_target = math.sqrt((red_pos[0] - green_center[0])**2 + 
                                             (red_pos[1] - green_center[1])**2)
                
                if distance_to_target < 0.1:  # 10cm from target center
                    print(f"Red object near target! Distance: {distance_to_target:.3f}m")
                    # Continue for a few more steps to see if it's stable
                    if step_idx > max_steps * 0.8:  # Only in later part of episode
                        print("Ending early - red object appears to be at target")
                        break
                
            except Exception as e:
                print(f"Error at step {step_idx}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    print(f"CNN control completed after {step_count} steps")
    
    # Wait for final settling
    step(int(1.0 / DT))
    
    # Final validation
    success, validation_info = validate_red_on_green(red_obj, green_center, green_size)
    
    print(f"\n--- CNN Control Results ---")
    if validation_info['success']:
        print("✓ SUCCESS: CNN successfully placed red object on green surface!")
        print(f"  Position offsets: x={validation_info['x_offset_from_center']:.3f}m, "
              f"y={validation_info['y_offset_from_center']:.3f}m")
    else:
        print("✗ FAILED: CNN did not successfully complete the task")
        print(f"  X in bounds: {validation_info['x_in_bounds']}")
        print(f"  Y in bounds: {validation_info['y_in_bounds']}")
        print(f"  Z correct: {validation_info['z_correct']}")
    
    print(f"Final red object position: [{validation_info['red_position'][0]:.3f}, "
          f"{validation_info['red_position'][1]:.3f}, {validation_info['red_position'][2]:.3f}]")
    
    # Save camera state for next episode
    if gui:
        save_camera_state()
    
    # Cleanup
    if video_recorder is not None:
        video_recorder.close()
    
    time.sleep(2.0 if gui else 0.5)  # Let user observe results
    p.disconnect()


# ------------------------------------------------------------
# Original Functions (kept from your original script)
# ------------------------------------------------------------
def load_initial_positions(csv_path: str) -> List[Dict]:
    """Load initial object positions from CSV file."""
    positions_info = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            positions_info.append({
                'object_id': int(row['object_id']),
                'body_id': int(row['body_id']),  # Will be replaced with new body IDs
                'color': row['color'],
                'x': float(row['x']),
                'y': float(row['y']),
                'z': float(row['z'])
            })
    return positions_info


def spawn_boxes_from_positions(positions_info: List[Dict]) -> Tuple[List[int], int]:
    """Spawn objects at specific positions loaded from CSV."""
    bodies = []
    red_body_id = None
    
    for obj_info in positions_info:
        pos = (obj_info['x'], obj_info['y'], obj_info['z'])
        color = RED if obj_info['color'] == 'red' else GRAY
        
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[OBJ_HALF] * 3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[OBJ_HALF] * 3, rgbaColor=color)
        b = p.createMultiBody(baseMass=OBJ_MASS, baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis, basePosition=list(pos))
        set_friction(b, -1, lateral=1.2)
        bodies.append(b)
        
        if obj_info['color'] == 'red':
            red_body_id = b
    
    return bodies, red_body_id


def save_initial_positions(ep_idx: int, positions_info: List[Dict]) -> None:
    """Save initial object positions to a CSV file."""
    ep_dir = os.path.join(DATA_ROOT, f"episode_{ep_idx:04d}")
    os.makedirs(ep_dir, exist_ok=True)
    
    csv_path = os.path.join(ep_dir, "initial_positions.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['object_id', 'body_id', 'color', 'x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for obj_info in positions_info:
            writer.writerow(obj_info)
    
    print(f"Initial positions saved to: {csv_path}")


def load_actions(csv_path: str) -> List[Dict]:
    """Load action sequence from CSV file."""
    actions = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            action = {
                'frame': int(row['frame']),
                'ee_tx': float(row['ee_tx']),
                'ee_ty': float(row['ee_ty']),
                'ee_tz': float(row['ee_tz']),
                'ee_qx': float(row['ee_qx']),
                'ee_qy': float(row['ee_qy']),
                'ee_qz': float(row['ee_qz']),
                'ee_qw': float(row['ee_qw']),
                'q0': float(row['q0']),
                'q1': float(row['q1']),
                'q2': float(row['q2']),
                'q3': float(row['q3']),
                'q4': float(row['q4']),
                'q5': float(row['q5']),
                'q6': float(row['q6']),
                'gripper_open_width': float(row['gripper_open_width'])
            }
            actions.append(action)
    return actions


def save_validation_result(ep_idx: int, validation_info: Dict) -> None:
    """Save validation results to a file."""
    ep_dir = os.path.join(DATA_ROOT, f"episode_{ep_idx:04d}")
    
    csv_path = os.path.join(ep_dir, "validation_result.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        for key, value in validation_info.items():
            if isinstance(value, tuple):
                if len(value) == 2:
                    value = f"({value[0]:.4f}, {value[1]:.4f})"
                elif len(value) == 3:
                    value = f"({value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f})"
                else:
                    value = str(value)
            elif isinstance(value, float):
                value = f"{value:.4f}"
            writer.writerow([key, value])
    
    if validation_info['success']:
        print(f"✓ Episode {ep_idx}: Validation SUCCESSFUL - Red object correctly placed on green surface")
        print(f"  Position offsets from center: x={validation_info['x_offset_from_center']:.3f}m, "
              f"y={validation_info['y_offset_from_center']:.3f}m")
    else:
        print(f"✗ Episode {ep_idx}: Validation FAILED - Red object NOT on green surface")
        print(f"  X in bounds: {validation_info['x_in_bounds']}, "
              f"Y in bounds: {validation_info['y_in_bounds']}, "
              f"Z correct: {validation_info['z_correct']}")


class DataRecorder:
    """Utility to record camera frames and action vectors for imitation learning."""

    def __init__(self, root: str, ep_idx: int, cameras_dict: dict, img_w: int = IMG_W, img_h: int = IMG_H, save_every: int = 1):
        self.ep_idx = ep_idx
        self.img_w = img_w
        self.img_h = img_h
        self.cameras_dict = cameras_dict
        self.frame = 0
        self.save_every = max(1, int(save_every))

        # Create directories
        self.base_dir = os.path.join(root, f"episode_{ep_idx:04d}")
        self.cam_dirs = {}
        for cam_name in cameras_dict.keys():
            cam_dir = os.path.join(self.base_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            self.cam_dirs[cam_name] = cam_dir

        # Open CSV for action logging
        self.csv_path = os.path.join(self.base_dir, "actions.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        
        header = ["frame"] + [f"{cam}_image" for cam in cameras_dict.keys()] + [
            "ee_tx", "ee_ty", "ee_tz",
            "ee_qx", "ee_qy", "ee_qz", "ee_qw",
            "q0","q1","q2","q3","q4","q5","q6",
            "gripper_open_width",
        ]
        self.writer.writerow(header)

    def _save_camera(self, view, proj, path: str) -> None:
        """Capture current scene with provided camera matrices and save PNG to path."""
        w, h, rgba, _, _ = p.getCameraImage(self.img_w, self.img_h, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgba = np.array(rgba, dtype=np.uint8)
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]
        Image.fromarray(rgb).save(path)

    def next_frame(self, action_vec: List[float]) -> None:
        """Capture all cameras and append the given action vector for the current frame."""
        should_save = (self.frame % self.save_every == 0)
        if should_save:
            image_paths = []
            for cam_name, (view, proj) in self.cameras_dict.items():
                img_name = f"{cam_name}_{self.frame:06d}.png"
                img_path = os.path.join(self.cam_dirs[cam_name], img_name)
                self._save_camera(view, proj, img_path)
                image_paths.append(os.path.relpath(img_path, self.base_dir))
            
            row = [self.frame] + image_paths + action_vec
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
    """Open/close the gripper fingers to a target opening width."""
    tgt = clamp(open_width * 0.5, 0.0, 0.045)
    for _ in range(speed_steps):
        p.setJointMotorControlArray(robot, finger_joints, p.POSITION_CONTROL,
                                    targetPositions=[tgt] * len(finger_joints),
                                    forces=[FINGER_FORCE] * len(finger_joints))
        step(1)
        if recorder is not None:
            if arm_joint_targets is None:
                arm_idxs = joint_indices(robot)
                arm_joint_targets_now = [p.getJointState(robot, j)[0] for j in arm_idxs]
            else:
                arm_joint_targets_now = arm_joint_targets

            if ee_target is None:
                ee = get_link_index_by_name(robot, "panda_hand")
                ls = p.getLinkState(robot, ee, computeForwardKinematics=True)
                ee_pos, ee_orn = ls[4], ls[5]
            else:
                ee_pos, ee_orn = ee_target

            action_vec = list(ee_pos) + list(ee_orn) + list(arm_joint_targets_now[:7]) + [open_width]
            recorder.next_frame(action_vec)


def move_ik(robot: int, arm_joints: List[int], ee_link: int, pos: List[float], orn: List[float], duration_s: float = 1.0,
            recorder: Optional[DataRecorder] = None, gripper_open_width: float = 0.08) -> List[float]:
    """Smoothly move the arm using IK to reach the target pose (pos, orn)."""
    steps = max(1, int(duration_s / DT))
    q_start = [p.getJointState(robot, j)[0] for j in arm_joints]
    
    q_goal = p.calculateInverseKinematics(
        robot, ee_link, pos, orn,
        restPoses=q_start,
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


def replay_action(robot: int, arm_joints: List[int], finger_joints: List[int], action: Dict, 
                  wait_time: float = REPLAY_STEP_TIME, max_wait_time: float = 0.2) -> None:
    """Execute a single action from the recorded sequence."""
    joint_targets = [action[f'q{i}'] for i in range(7)]
    gripper_width = action['gripper_open_width']
    
    p.setJointMotorControlArray(robot, arm_joints[:7], p.POSITION_CONTROL,
                               targetPositions=joint_targets, 
                               forces=[ARM_FORCE] * 7)
    
    gripper_target = clamp(gripper_width * 0.5, 0.0, 0.045)
    p.setJointMotorControlArray(robot, finger_joints, p.POSITION_CONTROL,
                               targetPositions=[gripper_target] * len(finger_joints),
                               forces=[FINGER_FORCE] * len(finger_joints))
    
    step_count = max(1, int(wait_time / DT))
    max_steps = max(step_count, int(max_wait_time / DT))
    
    for i in range(max_steps):
        step(1)
        if i >= step_count:
            break


# ------------------------------------------------------------
# Original Main Functions (kept from your script)
# ------------------------------------------------------------
def run_episode_record(ep_idx: int, gui: bool = True, seed: Optional[int] = None, 
                      save_every: int = 1, first_episode: bool = False) -> None:
    """Run a single recording episode: spawn objects, pick the red one, place on green surface, and record data."""
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
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

    # Green target surface
    green_center = (0.25, -0.25)
    green_size = (0.18, 0.14)
    _, drop_pos = create_green_target_surface(center=green_center, size=green_size)

    # Objects with spacing constraints (avoiding green surface)
    objs, red_obj, initial_positions = spawn_boxes_gray_and_one_red(
        N_OBJECTS, seed=seed,
        avoid_center=tuple(panda_base_pos),
        avoid_radius=RED_MIN_DIST_FROM_ROBOT,
        clearance_red=CLEARANCE_RED,
        clearance_any=CLEARANCE_ANY,
        green_surface_center=green_center,
        green_surface_size=green_size
    )
    
    save_initial_positions(ep_idx, initial_positions)

    # Wait for settling
    step(240)

    # Home pose to a comfortable middle configuration
    mid = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    p.setJointMotorControlArray(panda, arm[:7], p.POSITION_CONTROL, targetPositions=mid, forces=[ARM_FORCE] * 7)
    step(int(1.0 / DT))

    # Setup all cameras & recorder
    cameras_dict = setup_cameras()
    recorder = DataRecorder(DATA_ROOT, ep_idx, cameras_dict, IMG_W, IMG_H, save_every=save_every)

    # Open gripper (record action)
    set_gripper(panda, fingers, open_width=0.08, speed_steps=80, recorder=recorder,
                arm_joint_targets=mid, ee_target=(p.getLinkState(panda, ee)[4], p.getLinkState(panda, ee)[5]))

    # Determine target pose above red object
    pos_obj, _ = p.getBasePositionAndOrientation(red_obj)
    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    approach = [pos_obj[0], pos_obj[1], TABLE_TOP_Z + 0.18]
    descend  = [pos_obj[0], pos_obj[1], TABLE_TOP_Z + OBJ_HALF + 0.05]
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
    
    # Lower to place on green surface
    lower_drop = [drop_pos[0], drop_pos[1], drop_pos[2] + 0.05]
    q_last = move_ik(panda, arm, ee, lower_drop, down_orn, duration_s=0.8, recorder=recorder, gripper_open_width=0.0)

    # Open to release
    set_gripper(panda, fingers, open_width=0.08, speed_steps=80, recorder=recorder,
                arm_joint_targets=q_last, ee_target=(lower_drop, down_orn))
    if grasp_cid is not None:
        p.removeConstraint(grasp_cid)

    # Wait for object to settle briefly
    step(int(0.3 / DT))
    
    # Move arm up and away after releasing the object
    retreat_pos = [lower_drop[0], lower_drop[1], TABLE_TOP_Z + 0.20]
    q_last = move_ik(panda, arm, ee, retreat_pos, down_orn, duration_s=0.8, recorder=recorder, gripper_open_width=0.08)
    
    # Optional: Move to a safe home-like position
    safe_pos = [panda_base_pos[0] + 0.3, panda_base_pos[1], TABLE_TOP_Z + 0.25]
    q_last = move_ik(panda, arm, ee, safe_pos, down_orn, duration_s=1.0, recorder=recorder, gripper_open_width=0.08)

    # Wait for final settling and validation
    step(int(0.4 / DT))
    
    # Validate placement
    success, validation_info = validate_red_on_green(red_obj, green_center, green_size)
    save_validation_result(ep_idx, validation_info)

    # Save camera state before closing
    if gui:
        save_camera_state()

    # Cleanup
    recorder.close()
    time.sleep(0.2)
    p.disconnect()


def run_episode_replay(ep_idx: int, initial_positions_path: str, actions_path: str, 
                      gui: bool = True, replay_speed: float = 1.0, save_video: bool = False) -> None:
    """Replay a recorded episode by loading initial positions and action sequence."""
    print(f"\n--- Replaying Episode {ep_idx} ---")
    print(f"Initial positions: {initial_positions_path}")
    print(f"Actions: {actions_path}")
    
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

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

    # Green target surface
    green_center = (0.25, -0.25)
    green_size = (0.18, 0.14)
    _, drop_pos = create_green_target_surface(center=green_center, size=green_size)

    # Load and spawn objects from initial positions
    print("Loading initial positions...")
    initial_positions = load_initial_positions(initial_positions_path)
    objs, red_obj = spawn_boxes_from_positions(initial_positions)
    print(f"Spawned {len(objs)} objects (red object ID: {red_obj})")

    # Wait for settling
    step(240)

    # Load action sequence
    print("Loading action sequence...")
    actions = load_actions(actions_path)
    print(f"Loaded {len(actions)} actions")

    # Setup video recording if requested
    video_recorder = None
    if save_video:
        cameras_dict = setup_cameras()
        video_recorder = DataRecorder(DATA_ROOT, ep_idx, cameras_dict, IMG_W, IMG_H, save_every=1)

    # Calculate wait time based on replay speed
    action_wait_time = REPLAY_STEP_TIME / replay_speed

    # Replay actions
    print("Starting replay...")
    for i, action in enumerate(actions):
        if i % 100 == 0:
            print(f"Replaying action {i}/{len(actions)}")
        
        replay_action(panda, arm, fingers, action, wait_time=action_wait_time)
        
        if video_recorder is not None:
            action_vec = [action['ee_tx'], action['ee_ty'], action['ee_tz'],
                         action['ee_qx'], action['ee_qy'], action['ee_qz'], action['ee_qw']] + \
                        [action[f'q{j}'] for j in range(7)] + [action['gripper_open_width']]
            video_recorder.next_frame(action_vec)

    # Final validation
    success, validation_info = validate_red_on_green(red_obj, green_center, green_size)
    print(f"\nReplay validation result:")
    if validation_info['success']:
        print(f"✓ Replay SUCCESSFUL - Red object correctly placed on green surface")
        print(f"  Position offsets from center: x={validation_info['x_offset_from_center']:.3f}m, "
              f"y={validation_info['y_offset_from_center']:.3f}m")
    else:
        print(f"✗ Replay FAILED - Red object NOT on green surface")

    # Cleanup
    if video_recorder is not None:
        video_recorder.close()
    
    print("Replay complete!")
    time.sleep(0.2)
    p.disconnect()


def main_record(gui: bool = True, episodes: int = 3, seed: Optional[int] = 42, save_every: int = 1) -> None:
    """Main entry for recording mode: run multiple episodes of pick-and-place data collection."""
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting Pick-and-Place Data Collection with Validation")
    print("="*60)
    
    successful_episodes = 0
    
    for ep in range(episodes):
        print(f"\n--- Episode {ep}/{episodes-1} ---")
        run_seed = None if seed is None else (seed + ep)
        run_episode_record(ep, gui=gui, seed=run_seed, save_every=save_every, first_episode=(ep == 0))
        
        # Check if episode was successful
        validation_file = os.path.join(DATA_ROOT, f"episode_{ep:04d}", "validation_result.csv")
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == 'success' and row[1] == 'True':
                        successful_episodes += 1
                        break
    
    print("\n" + "="*60)
    print(f"Data Collection Complete!")
    print(f"Success Rate: {successful_episodes}/{episodes} ({100*successful_episodes/episodes:.1f}%)")
    print(f"Data saved in: {DATA_ROOT}/")
    print("="*60 + "\n")


def main_replay(episode_dir: str, gui: bool = True, replay_speed: float = 1.0, 
               save_video: bool = False, ep_idx: Optional[int] = None) -> None:
    """Main entry for replay mode: replay a recorded episode."""
    # Extract episode index if not provided
    if ep_idx is None:
        import re
        match = re.search(r'episode_(\d+)', episode_dir)
        ep_idx = int(match.group(1)) if match else 0
    
    # Check for required files
    initial_positions_path = os.path.join(episode_dir, "initial_positions.csv")
    actions_path = os.path.join(episode_dir, "actions.csv")
    
    if not os.path.exists(initial_positions_path):
        raise FileNotFoundError(f"Initial positions file not found: {initial_positions_path}")
    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"Actions file not found: {actions_path}")
    
    print("\n" + "="*60)
    print("Starting Episode Replay")
    print("="*60)
    
    run_episode_replay(ep_idx, initial_positions_path, actions_path, 
                      gui=gui, replay_speed=replay_speed, save_video=save_video)
    
    print("\n" + "="*60)
    print("Replay Complete!")
    print("="*60 + "\n")


def main_cnn_control(model_path: str, gui: bool = True, episodes: int = 5, 
                    seed: Optional[int] = 42, save_video: bool = False,
                    max_steps: int = CNN_CONTROL_STEPS) -> None:
    """Main entry for CNN control mode: use trained CNN to control robot on new object configurations.
    
    Args:
        model_path: path to the trained CNN model checkpoint.
        gui: whether to use PyBullet GUI.
        episodes: number of episodes to run with CNN control.
        seed: base random seed for object placement (each episode uses seed + ep_idx).
        save_video: whether to save video frames of CNN control.
        max_steps: maximum number of CNN control steps per episode.
    """
    if not TORCH_AVAILABLE:
        print("Error: PyTorch not available. CNN control mode requires PyTorch.")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    print("\n" + "="*60)
    print("Starting CNN Control Mode")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Save video: {save_video}")
    
    successful_episodes = 0
    
    for ep in range(episodes):
        print(f"\n--- CNN Control Episode {ep+1}/{episodes} ---")
        run_seed = None if seed is None else (seed + ep)
        
        try:
            run_episode_cnn_control(
                ep_idx=ep, 
                model_path=model_path,
                gui=gui,
                seed=run_seed,
                save_video=save_video,
                max_steps=max_steps
            )
            
            # Note: Success evaluation would need to be added to run_episode_cnn_control
            # For now, we assume success if no exception was raised
            successful_episodes += 1
            
        except Exception as e:
            print(f"Episode {ep} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"CNN Control Complete!")
    print(f"Completed Episodes: {successful_episodes}/{episodes}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka Robot Pick-and-Place: Record, Replay, or CNN Control")
    parser.add_argument("mode", choices=["record", "replay", "control"], 
                       help="Mode: record new episodes, replay existing ones, or use CNN control")
    
    # Recording arguments
    parser.add_argument("--episodes", type=int, default=3, 
                       help="Number of episodes to record/control (record/control mode)")
    parser.add_argument("--seed", type=int, default=123, 
                       help="Random seed for recording/control (record/control mode)")
    parser.add_argument("--save-every", type=int, default=50, 
                       help="Save every N-th frame (record mode)")
    
    # Replay arguments
    parser.add_argument("--episode-dir", type=str, 
                       help="Episode directory to replay (replay mode)")
    parser.add_argument("--replay-speed", type=float, default=1.0, 
                       help="Replay speed multiplier (replay mode)")
    
    # CNN Control arguments
    parser.add_argument("--model-path", type=str, 
                       help="Path to trained CNN model checkpoint (control mode)")
    parser.add_argument("--max-steps", type=int, default=CNN_CONTROL_STEPS,
                       help="Maximum CNN control steps per episode (control mode)")
    
    # Common arguments
    parser.add_argument("--save-video", action="store_true", 
                       help="Save video frames (replay/control mode)")
    parser.add_argument("--ep-idx", type=int, 
                       help="Episode index for naming (replay mode)")
    parser.add_argument("--gui", action="store_true", default=True, 
                       help="Use PyBullet GUI")
    parser.add_argument("--headless", action="store_true", 
                       help="Run without GUI (overrides --gui)")
    
    args = parser.parse_args()
    
    # Handle GUI setting
    gui = args.gui and not args.headless
    
    if args.mode == "record":
        main_record(gui=gui, episodes=args.episodes, seed=args.seed, save_every=args.save_every)
    
    elif args.mode == "replay":
        if not args.episode_dir:
            parser.error("--episode-dir is required for replay mode")
        main_replay(args.episode_dir, gui=gui, replay_speed=args.replay_speed, 
                   save_video=args.save_video, ep_idx=args.ep_idx)
    
    elif args.mode == "control":
        if not args.model_path:
            parser.error("--model-path is required for control mode")
        main_cnn_control(args.model_path, gui=gui, episodes=args.episodes,
                        seed=args.seed, save_video=args.save_video, max_steps=args.max_steps)

    # Example usage for direct execution:
    # Uncomment one of the following lines for testing:
    
    # Record mode example:
    # main_record(gui=True, episodes=2, seed=123, save_every=50)
    
    # Replay mode example:
    # main_replay("data/episode_0000", gui=True, replay_speed=1.0, save_video=False)
    
    # CNN Control mode example:
    # main_cnn_control("model_checkpoints/robot_cnn_best.pth", gui=True, episodes=3, seed=456)