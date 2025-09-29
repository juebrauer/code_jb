#!/usr/bin/env python3
"""
Kinematic Chain Control with Imitation Learning

This script implements three modes for learning kinematic control:
1. collect: Collect training data (image, action) pairs using demonstrator
2. train: Train CNN to map images to actions using collected data  
3. test: Use trained CNN to control the robotic arm

Modes:
- collect: Human demonstrator generates actions, data saved to filesystem
- train: CNN training with train/validation split for generalization testing
- test: Load trained CNN and use for arm control

Usage: 
  python script.py collect --num_dof 1 --data_dir ./training_data
  python script.py train --data_dir ./training_data --model_path ./model.pth
  python script.py test --model_path ./model.pth --num_dof 1

Author: Generated for kinematic chain imitation learning
Requirements: PySide6, torch, torchvision, numpy, opencv-python
"""

import sys
import os
import math
import random
import argparse
import numpy as np
import pickle
import json
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import time

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
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader, random_split
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some modes will not work.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Image processing may be limited.")


class KinematicChain:
    """
    Basic kinematic chain simulation.
    
    Provides core functionality for forward kinematics, action execution,
    and state management. Shared between collect and test modes.
    """
    
    def __init__(self, num_dof=1, arm_length=80):
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
        
        # Current target
        self.target_pos = (0, 0)
        self.target_tolerance = 15.0  # Pixels
        self.angle_step = math.radians(1)  # 1 degree steps
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        
        # Initialize positions and generate first target
        self.update_positions()
        self.generate_new_target()
    
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
    
    def generate_new_target(self):
        """
        Generate a new random target position within reachable workspace.
        """
        if self.num_dof == 1:
            # For 1 DOF: Target on the circle with radius = arm_length
            angle = random.uniform(0, 2 * math.pi)
            target_x = self.arm_length * math.cos(angle)
            target_y = self.arm_length * math.sin(angle)
            self.target_pos = (target_x, target_y)
        else:
            # For multi-DOF: Random position within 80% of max reach
            distance = random.uniform(0.2 * self.max_reach, 0.8 * self.max_reach)
            angle = random.uniform(0, 2 * math.pi)
            target_x = distance * math.cos(angle)
            target_y = distance * math.sin(angle)
            self.target_pos = (target_x, target_y)
    
    def reset_episode(self):
        """Reset for a new episode."""
        # Reset joint angles to random starting positions
        self.angles = [random.uniform(-math.pi, math.pi) for _ in range(self.num_dof)]
        
        # Update positions and generate new target
        self.update_positions()
        self.generate_new_target()
        
        # Reset counters
        self.step_count = 0
        self.episode_count += 1
    
    def get_distance_to_target(self) -> float:
        """
        Calculate Euclidean distance from end effector to target.
        
        Returns:
            float: Distance in pixels
        """
        dx = self.end_effector_pos[0] - self.target_pos[0]
        dy = self.end_effector_pos[1] - self.target_pos[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def is_target_reached(self) -> bool:
        """
        Check if the target has been reached within tolerance.
        
        Returns:
            bool: True if target is reached
        """
        return self.get_distance_to_target() <= self.target_tolerance
    
    
    def take_action(self, action: int):
        """
        Execute an action by adjusting joint angles.
        
        Action space: For each DOF i, actions are 2*i and 2*i+1
        - 2*i: Decrease angle by 1 degree
        - 2*i+1: Increase angle by 1 degree
        
        Args:
            action (int): Action index (0 to 2*num_dof-1)
        """
        if action < 0 or action >= 2 * self.num_dof:
            raise ValueError(f"Invalid action {action} for {self.num_dof} DOFs")
        
        # Decode action
        joint_index = action // 2
        action_type = action % 2
        
        # Apply action
        if action_type == 0:
            # Decrease angle by 1 degree
            self.angles[joint_index] -= self.angle_step
        else:  # action_type == 1
            # Increase angle by 1 degree
            self.angles[joint_index] += self.angle_step
        
        # Keep angles in reasonable range
        self.angles[joint_index] = ((self.angles[joint_index] + 2*math.pi) % (4*math.pi)) - 2*math.pi
        
        # Update positions
        self.update_positions()
        self.step_count += 1
        
        # Check if target reached for potential reset
        if self.is_target_reached():
            print(f"🎯 Target reached in {self.step_count} steps!")
            return True  # Episode done
        
        return False  # Episode continues
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space (2 actions per DOF)."""
        return 2 * self.num_dof


class ActionCNN(nn.Module):
    """
    CNN for imitation learning: maps images to action vectors.
    
    Architecture:
    - Convolutional layers for image processing
    - Fully connected layers for action prediction
    - Output layer with action probabilities/logits
    """
    
    def __init__(self, action_space_size: int):
        """
        Initialize the Action CNN.
        
        Args:
            action_space_size (int): Number of possible actions
        """
        super(ActionCNN, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # We'll calculate the actual size dynamically
        self.fc_input_size = None
        
        # Fully connected layers
        self.fc1 = None  # Will be initialized dynamically
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space_size)  # Output layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def _get_conv_output_size(self, input_shape, device):
        """Calculate the output size of convolutional layers."""
        dummy_input = torch.zeros(1, *input_shape).to(device)
        with torch.no_grad():
            x = self.relu(self.bn1(self.conv1(dummy_input)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Initialize fc1 on first forward pass
        if self.fc1 is None:
            device = x.device
            self.fc_input_size = self._get_conv_output_size(x.shape[1:], device)
            self.fc1 = nn.Linear(self.fc_input_size, 128).to(device)
            print(f"Initialized FC layer with input size: {self.fc_input_size}")
        
        # Convolutional layers with batch norm and ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ImitationAgent:
    """
    Agent for imitation learning using supervised learning on collected data.
    """
    
    def __init__(self, action_space_size: int, learning_rate: float = 1e-4):
        """
        Initialize the imitation learning agent.
        
        Args:
            action_space_size (int): Number of possible actions
            learning_rate (float): Learning rate for optimization
        """
        self.action_space_size = action_space_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network
        self.network = ActionCNN(action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        print(f"Imitation Agent initialized on {self.device}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for neural network input.
        
        Args:
            image (np.ndarray): Raw image array (H, W, C)
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for network
        """
        if CV2_AVAILABLE:
            # Resize image to 84x84
            image_resized = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
        else:
            # Fallback: simple downsample
            h, w = image.shape[:2]
            step_h, step_w = h // 84, w // 84
            image_resized = image[::step_h, ::step_w]
            if image_resized.shape[0] != 84 or image_resized.shape[1] != 84:
                image_resized = np.resize(image_resized, (84, 84, 3))
        
        # Convert to float and normalize to [0, 1]
        image_resized = image_resized.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        image_processed = np.transpose(image_resized, (2, 0, 1))
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(image_processed).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def select_action(self, state: torch.Tensor) -> int:
        """
        Select action using the trained network.
        
        Args:
            state (torch.Tensor): Current state (preprocessed image)
            
        Returns:
            int: Selected action
        """
        self.network.eval()
        with torch.no_grad():
            action_logits = self.network(state)
            return action_logits.argmax().item()
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'action_space_size': self.action_space_size
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        print(f"Model loaded from {filepath}")


class KinematicWidget(QWidget):
    """
    Visualization widget for kinematic chain.
    
    Provides rendering and image capture functionality.
    """
    
    def __init__(self, kinematic_chain):
        """
        Initialize the visualization widget.
        
        Args:
            kinematic_chain (KinematicChain): The kinematic chain instance
        """
        super().__init__()
        self.chain = kinematic_chain
        
        # Set fixed window size
        self.setFixedSize(400, 400)
        
        # Calculate center coordinates
        self.center_x = 200
        self.center_y = 200
        
        # Visual parameters
        self.joint_radius = 8
        self.arm_width = 4
        self.angle_line_length = 12
        self.end_effector_radius = 6
        self.target_size = 10  # Size of target X marker
        
    def paintEvent(self, event):
        """Draw the kinematic chain with target."""
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
        """Draw target as an X marker."""
        target_x = self.center_x + self.chain.target_pos[0]
        target_y = self.center_y + self.chain.target_pos[1]
        
        # Set pen for target (blue X)
        pen = QPen(QColor(0, 0, 255), 3)
        painter.setPen(pen)
        
        # Draw X marker
        half_size = self.target_size // 2
        painter.drawLine(
            int(target_x - half_size), int(target_y - half_size),
            int(target_x + half_size), int(target_y + half_size)
        )
        painter.drawLine(
            int(target_x - half_size), int(target_y + half_size),
            int(target_x + half_size), int(target_y - half_size)
        )
        
        # Draw target tolerance circle (light blue, dashed)
        pen = QPen(QColor(173, 216, 230), 1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush())  # No fill
        
        tolerance_diameter = int(2 * self.chain.target_tolerance)
        painter.drawEllipse(
            int(target_x - self.chain.target_tolerance),
            int(target_y - self.chain.target_tolerance),
            tolerance_diameter, tolerance_diameter
        )
    
    def _draw_info_text(self, painter):
        """Draw informational text."""
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        
        # Episode and performance info
        info_lines = [
            f"Episode: {self.chain.episode_count} | Step: {self.chain.step_count}",
            f"Success Rate: {self.chain.success_count}/{self.chain.episode_count if self.chain.episode_count > 0 else 1}",
            f"Distance to Target: {self.chain.get_distance_to_target():.1f}px",
            f"End Effector: ({self.chain.end_effector_pos[0]:.1f}, {self.chain.end_effector_pos[1]:.1f})",
            f"Target: ({self.chain.target_pos[0]:.1f}, {self.chain.target_pos[1]:.1f})"
        ]
        
        for i, line in enumerate(info_lines):
            painter.drawText(10, 20 + i * 15, line)
    
    def capture_image(self) -> np.ndarray:
        """
        Capture the current rendered scene as a numpy array for RL training.
        IMPORTANT: Captures only the visual scene WITHOUT text to prevent data leakage.
        
        Returns:
            np.ndarray: RGB image array (H, W, C) - pure visual data only
        """
        # Create a temporary widget for capturing clean image without text
        temp_widget = QWidget()
        temp_widget.setFixedSize(400, 400)
        
        # Set up painter for clean rendering
        pixmap = QPixmap(400, 400)
        pixmap.fill(QColor(255, 255, 255))  # White background
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw only the visual components (NO TEXT!)
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
        arr = np.array(ptr).reshape((height, width, 4))  # RGBA
        rgb_array = arr[:, :, :3].copy()  # Extract RGB only
        
        return rgb_array


class DataCollectionDataset(Dataset):
    """
    Dataset class for loading collected training data.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize dataset from collected data directory.
        
        Args:
            data_dir (str): Directory containing collected data
        """
        self.data_dir = Path(data_dir)
        self.data_files = list(self.data_dir.glob("*.pkl"))
        
        if not self.data_files:
            raise ValueError(f"No data files found in {data_dir}")
        
        print(f"Found {len(self.data_files)} data files")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load data sample
        with open(self.data_files[idx], 'rb') as f:
            sample = pickle.load(f)
        
        image = sample['image']
        action = sample['action']
        
        # Preprocess image
        if CV2_AVAILABLE:
            image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
        else:
            h, w = image.shape[:2]
            step_h, step_w = h // 84, w // 84
            image = image[::step_h, ::step_w]
            if image.shape[0] != 84 or image.shape[1] != 84:
                image = np.resize(image, (84, 84, 3))
        
        # Normalize and convert to CHW format
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return torch.FloatTensor(image), torch.LongTensor([action])


class MainWindow(QMainWindow):
    """
    Main window for kinematic chain simulation.
    
    Supports collect and test modes.
    """
    
    def __init__(self, mode: str, num_dof=1, **kwargs):
        """
        Initialize the main window.
        
        Args:
            mode (str): Operation mode ('collect' or 'test')
            num_dof (int): Number of degrees of freedom
            **kwargs: Additional mode-specific arguments
        """
        super().__init__()
        
        self.mode = mode
        self.num_dof = num_dof
        
        # Configure window
        self.setWindowTitle(f"Kinematic Chain - {mode.title()} Mode - {num_dof} DOF")
        self.setFixedSize(420, 550)
        
        # Create kinematic chain
        self.chain = KinematicChain(num_dof)
        
        # Mode-specific initialization
        if mode == 'collect':
            self.data_dir = kwargs.get('data_dir', './training_data')
            self.data_counter = 0
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Data collection mode - saving to {self.data_dir}")
        
        elif mode == 'test':
            model_path = kwargs.get('model_path')
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            
            if not TORCH_AVAILABLE:
                raise ValueError("PyTorch not available - test mode requires PyTorch")
            
            self.agent = ImitationAgent(self.chain.get_action_space_size())
            self.agent.load_model(model_path)
            print(f"Test mode - loaded model from {model_path}")
        
        # Set up UI
        self._setup_ui()
        
        # Set up simulation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        self.timer.start(100)  # 10 FPS
    
    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Kinematic visualization
        self.kinematic_widget = KinematicWidget(self.chain)
        layout.addWidget(self.kinematic_widget)
        
        # Status information
        info_layout = QHBoxLayout()
        
        self.mode_label = QLabel(f"Mode: {self.mode.title()}")
        self.status_label = QLabel("Status: Running")
        
        if self.mode == 'collect':
            self.data_count_label = QLabel("Samples: 0")
            info_layout.addWidget(self.data_count_label)
        
        info_layout.addWidget(self.mode_label)
        info_layout.addWidget(self.status_label)
        
        layout.addLayout(info_layout)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def rl_step(self):
        """Perform one step of RL training with maximum speed."""
        # Performance monitoring
        import time
        current_time = time.time()
        if current_time - self.last_time >= 1.0:  # Every second
            self.steps_per_second = self.step_counter / (current_time - self.last_time)
            self.step_counter = 0
            self.last_time = current_time
            if self.training_enabled:
                print(f"Training speed: {self.steps_per_second:.1f} steps/sec, Epsilon: {self.agent.epsilon:.3f}")
        
        self.step_counter += 1
        
        if not self.training_enabled:
            # Demo mode: just show random actions
            action = random.randint(0, self.chain.get_action_space_size() - 1)
            reward, done = self.chain.take_action(action)
            
            if done:
                self.chain.reset_episode()
            
            self.kinematic_widget.update()
            return
        
        # Capture current state (image)
        current_image = self.kinematic_widget.capture_image()
        current_state = self.agent.preprocess_image(current_image)
        
        # Select and execute action
        action = self.agent.select_action(current_state)
        reward, done = self.chain.take_action(action)
        
        # Capture next state
        next_image = self.kinematic_widget.capture_image()
        next_state = self.agent.preprocess_image(next_image)
        
        # Store experience
        if self.current_state is not None:
            self.agent.store_experience(
                self.current_state, action, reward, next_state, done
            )
        
        # Train the agent (only occasionally to not slow down data collection)
        if self.step_counter % 4 == 0:  # Train every 4 steps instead of every step
            self.agent.train()
        
        # Update state
        self.current_state = next_state
        
        # Reset episode if done
        if done:
            self.chain.reset_episode()
            self.current_state = None
        
        # Update UI less frequently for better performance
        if self.step_counter % 10 == 0:  # Update display every 10 steps
            self.kinematic_widget.update()
            # Update epsilon display
            self.epsilon_label.setText(f"Epsilon: {self.agent.epsilon:.3f} | Speed: {self.steps_per_second:.0f} steps/sec")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep RL for Kinematic Chain Control",
        epilog="Example: python kinematic_chain_rl.py 1"
    )
    parser.add_argument(
        "num_dof", 
        nargs="?", 
        type=int, 
        default=1,
        help="Number of degrees of freedom. Start with 1 for initial experiments. Default: 1"
    )
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    if args.num_dof < 1:
        print("Error: Number of DOF must be at least 1.")
        return 1
    
    if args.num_dof > 3:
        print("Warning: Starting with more than 3 DOF may be challenging for initial RL experiments.")
        print("Consider starting with 1 DOF and gradually increasing complexity.")
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindowRL(args.num_dof)
    window.show()
    
    # Print startup information
    print(f"\n🤖 Deep RL Kinematic Chain Started")
    print(f"   • Degrees of Freedom: {args.num_dof}")
    print(f"   • Action Space Size: {2 * args.num_dof}")
    print(f"   • PyTorch Available: {TORCH_AVAILABLE}")
    print(f"   • Training: {'Enabled' if TORCH_AVAILABLE else 'Demo Mode'}")
    print(f"\n🎯 RL Setup:")
    print(f"   • Input: 800x800 RGB images")
    print(f"   • Actions: ±1° per DOF")
    print(f"   • Target: Random positions within reach")
    print(f"   • Success: Reach within 15px tolerance")
    print(f"\n📊 Visual Legend:")
    print(f"   • Black circles: Joints")
    print(f"   • Gray lines: Arms")
    print(f"   • Red circle: End effector")
    print(f"   • Blue X: Target position")
    print(f"   • Dashed circle: Target tolerance")
    
    if not TORCH_AVAILABLE:
        print(f"\n⚠️  Install PyTorch for RL training: pip install torch torchvision")
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())