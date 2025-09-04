#!/usr/bin/env python3
"""
Kinematic Chain Control with Collect/Train/Test Modes

This script provides three modes:
1. collect: Generate and save training data using expert demonstrations
2. train: Train a CNN on collected data with validation
3. test: Load trained CNN and use it to control the kinematic chain

Usage:
    python kinematic_chain_modes.py collect --dof 2 --samples 1000
    python kinematic_chain_modes.py train --dof 2 --epochs 100
    python kinematic_chain_modes.py test --dof 2 --model path/to/model.pth

Author: Generated for modular kinematic chain learning
Requirements: PySide6, torch, torchvision, numpy, opencv-python
"""

import sys
import os
import math
import random
import argparse
import numpy as np
from typing import Tuple, List, Optional
import time
import json
import pickle
from pathlib import Path

# Qt imports
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap
from PySide6.QtCore import Qt

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import cv2
    from torch.utils.data import Dataset, DataLoader, random_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/OpenCV/sklearn/matplotlib not available.")


class KinematicChain:
    """
    Base kinematic chain class for all modes.
    
    Handles forward kinematics, target generation, and basic control.
    """
    
    def __init__(self, num_dof=2, arm_length=80):
        """
        Initialize the kinematic chain.
        
        Args:
            num_dof (int): Number of degrees of freedom
            arm_length (float): Length of each arm segment in pixels
        """
        self.num_dof = num_dof
        self.arm_length = arm_length
        self.max_reach = num_dof * arm_length
        
        # Initialize joint angles
        self.angles = [0.0 for _ in range(num_dof)]
        
        # Position tracking
        self.joint_positions = [(0, 0)]
        self.end_effector_pos = (0, 0)
        
        # Target and tolerance
        self.target_pos = (0, 0)
        self.target_tolerance = 15.0
        self.angle_step = math.radians(1)  # 1 degree steps
        
        # Scenario tracking
        self.scenario_count = 0
        self.current_step = 0
        self.scenario_complete = False
        
        # Initialize positions and generate first target
        self.update_positions()
        self.generate_new_scenario()
    
    def update_positions(self):
        """Calculate all joint positions using forward kinematics."""
        self.joint_positions = [(0, 0)]
        
        x, y = 0, 0
        cumulative_angle = 0
        
        for i in range(self.num_dof):
            cumulative_angle += self.angles[i]
            x += self.arm_length * math.cos(cumulative_angle)
            y += self.arm_length * math.sin(cumulative_angle)
            self.joint_positions.append((x, y))
        
        self.end_effector_pos = (x, y)
    
    def generate_new_scenario(self):
        """Generate a new random scenario."""
        # Random starting configuration
        self.angles = [random.uniform(-math.pi, math.pi) for _ in range(self.num_dof)]
        self.update_positions()
        
        # Generate reachable target
        self.target_pos = self.generate_reachable_target()
        
        # Reset scenario tracking
        self.scenario_count += 1
        self.current_step = 0
        self.scenario_complete = False
        
        print(f"Scenario {self.scenario_count}: New target generated")
        print(f"  Start EE: ({self.end_effector_pos[0]:.1f}, {self.end_effector_pos[1]:.1f})")
        print(f"  Target: ({self.target_pos[0]:.1f}, {self.target_pos[1]:.1f})")
        print(f"  Initial distance: {self.get_distance_to_target():.1f}px")
    
    def generate_reachable_target(self) -> Tuple[float, float]:
        """Generate a target that is definitely reachable."""
        if self.num_dof == 1:
            # For 1 DOF: Target MUST be exactly on the circle
            angle = random.uniform(0, 2 * math.pi)
            target_x = self.arm_length * math.cos(angle)
            target_y = self.arm_length * math.sin(angle)
            return (target_x, target_y)
        else:
            # For multiple DOFs: Generate within conservative workspace bounds
            max_distance = 0.7 * self.max_reach
            min_distance = 0.2 * self.max_reach
            
            distance = random.uniform(min_distance, max_distance)
            angle = random.uniform(0, 2 * math.pi)
            
            target_x = distance * math.cos(angle)
            target_y = distance * math.sin(angle)
            
            return (target_x, target_y)
    
    def get_distance_to_target(self) -> float:
        """Calculate distance from end effector to target."""
        dx = self.end_effector_pos[0] - self.target_pos[0]
        dy = self.end_effector_pos[1] - self.target_pos[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def is_target_reached(self) -> bool:
        """Check if target is reached within tolerance."""
        return self.get_distance_to_target() <= self.target_tolerance
    
    def get_expert_action(self) -> int:
        """
        Generate expert action using optimal inverse kinematics.
        
        Returns:
            int: Best action (DOF_i * 3 + direction)
                 direction: 0=no_change, 1=-1°, 2=+1°
        """
        if self.is_target_reached():
            return 0  # DOF 0, no change
        
        if self.num_dof == 1:
            # For 1 DOF: Use analytical solution
            target_x, target_y = self.target_pos
            target_angle = math.atan2(target_y, target_x)
            current_angle = self.angles[0]
            
            # Calculate angular difference
            angle_diff = target_angle - current_angle
            
            # Normalize angle difference to [-π, π]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Choose direction based on shortest path
            if abs(angle_diff) < math.radians(0.5):
                return 0  # No change needed
            elif angle_diff > 0:
                return 2  # Turn right (+1°)
            else:
                return 1  # Turn left (-1°)
        
        else:
            # For multi-DOF: Use greedy search
            best_action = 0
            best_distance = self.get_distance_to_target()
            
            # Test all possible single-DOF movements
            for dof in range(self.num_dof):
                for direction in [1, 2]:  # Only test -1° and +1°
                    # Save current state
                    original_angle = self.angles[dof]
                    
                    # Try the action
                    if direction == 1:
                        self.angles[dof] -= self.angle_step
                    else:  # direction == 2
                        self.angles[dof] += self.angle_step
                    
                    # Update positions and check distance
                    self.update_positions()
                    new_distance = self.get_distance_to_target()
                    
                    # Is this better?
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_action = dof * 3 + direction
                    
                    # Restore original state
                    self.angles[dof] = original_angle
                    self.update_positions()
            
            return best_action
    
    def take_action(self, action: int) -> bool:
        """
        Execute an action and return whether scenario is complete.
        
        Args:
            action (int): Action to execute (DOF_i * 3 + direction)
            
        Returns:
            bool: True if scenario is complete
        """
        # Decode action
        joint_index = action // 3
        direction = action % 3
        
        # Apply action
        if direction == 0:
            pass  # No change
        elif direction == 1:
            self.angles[joint_index] -= self.angle_step
        else:  # direction == 2
            self.angles[joint_index] += self.angle_step
        
        # Keep angles in reasonable range
        if joint_index < len(self.angles):
            self.angles[joint_index] = ((self.angles[joint_index] + 2*math.pi) % (4*math.pi)) - 2*math.pi
        
        self.update_positions()
        self.current_step += 1
        
        # Check if scenario is complete
        if self.is_target_reached():
            self.scenario_complete = True
            print(f"  Scenario {self.scenario_count} completed in {self.current_step} steps!")
            return True
        
        # Prevent infinite scenarios
        if self.current_step > 100:
            self.scenario_complete = True
            print(f"  Scenario {self.scenario_count} timed out after {self.current_step} steps")
            return True
        
        return False
    
    def get_action_space_size(self) -> int:
        """Get total number of possible actions."""
        return 3 * self.num_dof


class ActionCNN(nn.Module):
    """
    Convolutional Neural Network for action prediction.
    
    Maps images to action probabilities.
    """
    
    def __init__(self, action_space_size: int):
        """
        Initialize the CNN.
        
        Args:
            action_space_size (int): Number of possible actions
        """
        super(ActionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate feature size dynamically
        self.fc_input_size = None
        
        # Fully connected layers
        self.fc1 = None  # Initialized dynamically
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space_size)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def _get_conv_output_size(self, input_shape, device):
        """Calculate convolutional output size."""
        dummy_input = torch.zeros(1, *input_shape).to(device)
        with torch.no_grad():
            x = self.relu(self.conv1(dummy_input))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Initialize fc1 on first forward pass
        if self.fc1 is None:
            device = x.device
            self.fc_input_size = self._get_conv_output_size(x.shape[1:], device)
            self.fc1 = nn.Linear(self.fc_input_size, 128).to(device)
            print(f"Initialized CNN with conv output size: {self.fc_input_size}")
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class KinematicDataset(Dataset):
    """Dataset for loading kinematic chain training data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize dataset.
        
        Args:
            data_dir (str): Directory containing training data
        """
        self.data_dir = Path(data_dir)
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.num_samples = self.metadata['total_samples']
        print(f"Loaded dataset with {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Load image
        image_path = self.data_dir / f"image_{idx:06d}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        image = cv2.resize(image, (84, 84))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW format
        
        # Load action
        action_path = self.data_dir / f"action_{idx:06d}.txt"
        with open(action_path, 'r') as f:
            action = int(f.read().strip())
        
        return torch.FloatTensor(image), torch.LongTensor([action])[0]


class KinematicWidget(QWidget):
    """Visualization widget for kinematic chain."""
    
    def __init__(self, chain):
        """
        Initialize the visualization widget.
        
        Args:
            chain (KinematicChain): The kinematic chain
        """
        super().__init__()
        self.chain = chain
        
        # Window parameters
        self.setFixedSize(400, 400)
        self.center_x = 200
        self.center_y = 200
        
        # Visual parameters
        self.joint_radius = 8
        self.arm_width = 4
        self.angle_line_length = 12
        self.end_effector_radius = 6
        self.target_size = 10
    
    def paintEvent(self, event):
        """Draw the kinematic chain."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # Draw components
        self._draw_arms(painter)
        self._draw_joints(painter)
        self._draw_end_effector(painter)
        self._draw_target(painter)
        self._draw_info_text(painter)
    
    def _draw_arms(self, painter):
        """Draw arm segments."""
        pen = QPen(QColor(128, 128, 128), self.arm_width)
        painter.setPen(pen)
        
        for i in range(len(self.chain.joint_positions) - 1):
            start_x = self.center_x + self.chain.joint_positions[i][0]
            start_y = self.center_y + self.chain.joint_positions[i][1]
            end_x = self.center_x + self.chain.joint_positions[i + 1][0]
            end_y = self.center_y + self.chain.joint_positions[i + 1][1]
            
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
    
    def _draw_joints(self, painter):
        """Draw joint circles and angle indicators."""
        brush = QBrush(QColor(0, 0, 0))
        painter.setBrush(brush)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        
        for i, (x, y) in enumerate(self.chain.joint_positions[:-1]):
            joint_x = self.center_x + x
            joint_y = self.center_y + y
            
            painter.drawEllipse(
                int(joint_x - self.joint_radius), 
                int(joint_y - self.joint_radius),
                2 * self.joint_radius, 
                2 * self.joint_radius
            )
            
            if i < len(self.chain.angles):
                self._draw_angle_indicator(painter, joint_x, joint_y, i)
    
    def _draw_angle_indicator(self, painter, joint_x, joint_y, joint_index):
        """Draw angle indicator line."""
        cumulative_angle = sum(self.chain.angles[:joint_index + 1])
        
        line_end_x = joint_x + self.angle_line_length * math.cos(cumulative_angle)
        line_end_y = joint_y + self.angle_line_length * math.sin(cumulative_angle)
        
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        
        painter.drawLine(
            int(joint_x), int(joint_y), 
            int(line_end_x), int(line_end_y)
        )
    
    def _draw_end_effector(self, painter):
        """Draw end effector."""
        brush = QBrush(QColor(255, 0, 0))
        painter.setBrush(brush)
        painter.setPen(QPen(QColor(255, 0, 0), 1))
        
        ee_x = self.center_x + self.chain.end_effector_pos[0]
        ee_y = self.center_y + self.chain.end_effector_pos[1]
        
        painter.drawEllipse(
            int(ee_x - self.end_effector_radius), 
            int(ee_y - self.end_effector_radius),
            2 * self.end_effector_radius, 
            2 * self.end_effector_radius
        )
    
    def _draw_target(self, painter):
        """Draw target as X marker."""
        target_x = self.center_x + self.chain.target_pos[0]
        target_y = self.center_y + self.chain.target_pos[1]
        
        # Draw blue X
        pen = QPen(QColor(0, 0, 255), 3)
        painter.setPen(pen)
        
        half_size = self.target_size // 2
        painter.drawLine(
            int(target_x - half_size), int(target_y - half_size),
            int(target_x + half_size), int(target_y + half_size)
        )
        painter.drawLine(
            int(target_x - half_size), int(target_y + half_size),
            int(target_x + half_size), int(target_y - half_size)
        )
        
        # Draw tolerance circle
        pen = QPen(QColor(173, 216, 230), 1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush())
        
        tolerance_diameter = int(2 * self.chain.target_tolerance)
        painter.drawEllipse(
            int(target_x - self.chain.target_tolerance),
            int(target_y - self.chain.target_tolerance),
            tolerance_diameter, tolerance_diameter
        )
    
    def _draw_info_text(self, painter):
        """Draw informational text."""
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        
        info_lines = [
            f"Scenario: {self.chain.scenario_count} | Step: {self.chain.current_step}",
            f"Distance to Target: {self.chain.get_distance_to_target():.1f}px",
            f"End Effector: ({self.chain.end_effector_pos[0]:.1f}, {self.chain.end_effector_pos[1]:.1f})",
            f"Target: ({self.chain.target_pos[0]:.1f}, {self.chain.target_pos[1]:.1f})"
        ]
        
        for i, line in enumerate(info_lines):
            painter.drawText(10, 20 + i * 15, line)
    
    def capture_image(self) -> np.ndarray:
        """
        Capture clean image without text for learning.
        
        Returns:
            np.ndarray: RGB image array (H, W, C)
        """
        pixmap = QPixmap(400, 400)
        pixmap.fill(QColor(255, 255, 255))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw only visual components (no text)
        self._draw_arms(painter)
        self._draw_joints(painter)
        self._draw_end_effector(painter)
        self._draw_target(painter)
        
        painter.end()
        
        # Convert to numpy array
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        
        ptr = image.constBits()
        arr = np.array(ptr).reshape((height, width, 4))
        rgb_array = arr[:, :, :3].copy()
        
        return rgb_array


class MainWindow(QMainWindow):
    """Main window for kinematic chain visualization."""
    
    def __init__(self, chain, mode, action_generator=None):
        """
        Initialize the main window.
        
        Args:
            chain (KinematicChain): The kinematic chain
            mode (str): Operating mode ('collect' or 'test')
            action_generator (callable): Function to generate actions
        """
        super().__init__()
        
        self.chain = chain
        self.mode = mode
        self.action_generator = action_generator
        
        # Configure window
        self.setWindowTitle(f"Kinematic Chain - {mode.upper()} Mode - {chain.num_dof} DOF")
        self.setFixedSize(420, 500)
        
        # Set up UI
        self._setup_ui()
        
        # Control variables
        self.steps_per_second = 0
        self.last_time = time.time()
        self.step_counter = 0
        
        # Data collection (for collect mode)
        if mode == 'collect':
            self.collected_samples = 0
            self.data_dir = None
            self.target_samples = None  # Will be set from collect_mode
        
        # Set up timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.timer.start(50)  # 20 FPS
        
        print(f"{mode.upper()} mode started with {chain.num_dof} DOF")
    
    def set_data_directory(self, data_dir: str):
        """Set data directory for collect mode."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_ui(self):
        """Set up user interface."""
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Kinematic visualization
        self.kinematic_widget = KinematicWidget(self.chain)
        layout.addWidget(self.kinematic_widget)
        
        # Status information
        info_layout = QVBoxLayout()
        
        self.mode_label = QLabel(f"Mode: {self.mode.upper()}")
        self.stats_label = QLabel("Statistics: Starting...")
        self.performance_label = QLabel("Performance: Initializing...")
        
        info_layout.addWidget(self.mode_label)
        info_layout.addWidget(self.stats_label)
        info_layout.addWidget(self.performance_label)
        
        layout.addLayout(info_layout)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def step(self):
        """Perform one step of the process."""
        # Performance monitoring
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.steps_per_second = self.step_counter / (current_time - self.last_time)
            self.step_counter = 0
            self.last_time = current_time
        
        self.step_counter += 1
        
        # Capture current image
        current_image = self.kinematic_widget.capture_image()
        
        # Generate action using provided action generator
        if self.action_generator:
            action = self.action_generator(current_image)
        else:
            action = 0  # Default: no action
        
        # Save data in collect mode
        if self.mode == 'collect' and self.data_dir:
            self._save_sample(current_image, action)
            
            # CHECK IF WE HAVE COLLECTED ENOUGH SAMPLES
            if self.target_samples and self.collected_samples >= self.target_samples:
                self.timer.stop()  # Stop the timer
                self._finalize_collection()  # Finalize and exit
                return
        
        # Execute action
        done = self.chain.take_action(action)
        
        if done:
            self.chain.generate_new_scenario()
        
        # Update UI
        self._update_status()
        self.kinematic_widget.update()
    
    def _save_sample(self, image: np.ndarray, action: int):
        """Save a training sample to disk."""
        # Save image
        image_path = self.data_dir / f"image_{self.collected_samples:06d}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save action
        action_path = self.data_dir / f"action_{self.collected_samples:06d}.txt"
        with open(action_path, 'w') as f:
            f.write(str(action))
        
        self.collected_samples += 1
    
    def _finalize_collection(self):
        """Finalize the collection process and save metadata."""
        print(f"\nCollection complete! Saved exactly {self.collected_samples} samples.")
        
        # Save metadata
        metadata = {
            'dof': self.chain.num_dof,
            'total_samples': self.collected_samples,
            'action_space_size': self.chain.get_action_space_size(),
            'arm_length': self.chain.arm_length,
            'target_tolerance': self.chain.target_tolerance,
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
        
        # Close the application
        QApplication.instance().quit()
    
    def _update_status(self):
        """Update status labels."""
        mode_text = f"Mode: {self.mode.upper()} | Speed: {self.steps_per_second:.0f} steps/sec"
        self.mode_label.setText(mode_text)
        
        if self.mode == 'collect':
            if self.target_samples:
                stats_text = f"Samples collected: {self.collected_samples}/{self.target_samples} | Scenarios: {self.chain.scenario_count}"
            else:
                stats_text = f"Samples collected: {self.collected_samples} | Scenarios: {self.chain.scenario_count}"
            perf_text = f"Distance: {self.chain.get_distance_to_target():.1f}px | Current step: {self.chain.current_step}"
        else:  # test mode
            stats_text = f"Scenarios tested: {self.chain.scenario_count} | Current step: {self.chain.current_step}"
            perf_text = f"Distance: {self.chain.get_distance_to_target():.1f}px | AI controlling"
        
        self.stats_label.setText(stats_text)
        self.performance_label.setText(perf_text)


def collect_mode(args):
    """Run data collection mode."""
    print(f"\n=== COLLECT MODE ===")
    print(f"DOF: {args.dof}")
    print(f"Target samples: {args.samples}")
    print(f"Output directory: {args.output}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create kinematic chain
    chain = KinematicChain(args.dof)
    
    # Create action generator (expert)
    def expert_action_generator(image):
        return chain.get_expert_action()
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and configure main window
    window = MainWindow(chain, 'collect', expert_action_generator)
    window.set_data_directory(args.output)
    window.target_samples = args.samples  # SET TARGET NUMBER OF SAMPLES
    window.show()
    
    return app.exec()


def train_mode(args):
    """Run training mode."""
    if not TORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install torch and related packages.")
        return 1
    
    print(f"\n=== TRAIN MODE ===")
    print(f"DOF: {args.dof}")
    print(f"Data directory: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Load dataset
    try:
        dataset = KinematicDataset(args.data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Check DOF consistency
    metadata_path = Path(args.data) / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if metadata['dof'] != args.dof:
        print(f"Error: DOF mismatch. Data was collected with {metadata['dof']} DOF, but training with {args.dof} DOF.")
        return 1
    
    action_space_size = metadata['action_space_size']
    print(f"Action space size: {action_space_size}")
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = ActionCNN(action_space_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nStarting training...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_idx, (images, actions) in enumerate(train_loader):
            images, actions = images.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(1)
            epoch_train_total += actions.size(0)
            epoch_train_correct += predicted.eq(actions).sum().item()
        
        # Calculate training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100.0 * epoch_train_correct / epoch_train_total
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        
        with torch.no_grad():
            for images, actions in val_loader:
                images, actions = images.to(device), actions.to(device)
                outputs = model(images)
                loss = criterion(outputs, actions)
                
                epoch_val_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_val_total += actions.size(0)
                epoch_val_correct += predicted.eq(actions).sum().item()
        
        # Calculate validation metrics
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = 100.0 * epoch_val_correct / epoch_val_total
        
        # Store history
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch [{epoch+1}/{args.epochs}]")
            print(f"  Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
            print(f"  Val   - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    
    # Save model
    model_path = Path(args.output) / f"model_dof{args.dof}_epoch{args.epochs}.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'action_space_size': action_space_size,
        'dof': args.dof,
        'training_metadata': metadata,
        'training_history': {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plot_path = Path(args.output) / f"training_history_dof{args.dof}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training plot saved to: {plot_path}")
    
    # Final results
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Val Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Best Val Accuracy: {max(val_accuracies):.2f}%")
    
    return 0


def test_mode(args):
    """Run test mode."""
    if not TORCH_AVAILABLE:
        print("Error: PyTorch not available. Please install torch and related packages.")
        return 1
    
    print(f"\n=== TEST MODE ===")
    print(f"DOF: {args.dof}")
    print(f"Model: {args.model}")
    
    # Load model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model, map_location=device)
        
        # Check DOF consistency
        if checkpoint['dof'] != args.dof:
            print(f"Error: DOF mismatch. Model was trained with {checkpoint['dof']} DOF, but testing with {args.dof} DOF.")
            return 1
        
        # Create and load model
        model = ActionCNN(checkpoint['action_space_size']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Action space size: {checkpoint['action_space_size']}")
        print(f"Training metadata: {checkpoint['training_metadata']['collection_date']}")
        
    except FileNotFoundError:
        print(f"Error: Model file not found: {args.model}")
        return 1
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Create kinematic chain
    chain = KinematicChain(args.dof)
    
    # Create neural network action generator
    def nn_action_generator(image):
        # Preprocess image
        image_resized = cv2.resize(image, (84, 84))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = torch.FloatTensor(image_chw).unsqueeze(0).to(device)
        
        # Predict action
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_action = outputs.argmax().item()
        
        return predicted_action
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow(chain, 'test', nn_action_generator)
    window.show()
    
    print(f"Test mode running. Neural network is controlling the arm.")
    return app.exec()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kinematic Chain Control with Collect/Train/Test Modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 1000 training samples with 2 DOF
  python kinematic_chain_modes.py collect --dof 2 --samples 1000 --output data/dof2_1k

  # Train model on collected data
  python kinematic_chain_modes.py train --dof 2 --data data/dof2_1k --epochs 100 --output models/

  # Test trained model
  python kinematic_chain_modes.py test --dof 2 --model models/model_dof2_epoch100.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')
    
    # Collect mode
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('--dof', type=int, default=2, help='Number of degrees of freedom (default: 2)')
    collect_parser.add_argument('--samples', type=int, default=1000, help='Number of samples to collect (default: 1000)')
    collect_parser.add_argument('--output', type=str, default='data/training_data', help='Output directory (default: data/training_data)')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train neural network')
    train_parser.add_argument('--dof', type=int, default=2, help='Number of degrees of freedom (default: 2)')
    train_parser.add_argument('--data', type=str, required=True, help='Training data directory')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    train_parser.add_argument('--output', type=str, default='models/', help='Output directory for saved model (default: models/)')
    
    # Test mode
    test_parser = subparsers.add_parser('test', help='Test trained model')
    test_parser.add_argument('--dof', type=int, default=2, help='Number of degrees of freedom (default: 2)')
    test_parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
    
    return args


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    if args.dof < 1:
        print("Error: Number of DOF must be at least 1.")
        return 1
    
    # Print startup information
    print(f"\nKinematic Chain Control - {args.mode.upper()} Mode")
    print(f"Degrees of Freedom: {args.dof}")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    
    # Run appropriate mode
    if args.mode == 'collect':
        return collect_mode(args)
    elif args.mode == 'train':
        return train_mode(args)
    elif args.mode == 'test':
        return test_mode(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        return 1


if __name__ == "__main__":
    sys.exit(main())