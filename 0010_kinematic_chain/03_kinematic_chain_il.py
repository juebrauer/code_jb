#!/usr/bin/env python3
"""
Imitation Learning for Kinematic Chain Control

This script uses imitation learning to teach a CNN to control a kinematic chain.
The system generates expert demonstrations and trains a neural network to mimic them.

Features:
- Automatic expert demonstration generation
- Sequential single-DOF movements for simplicity
- Image-based supervised learning
- Configurable number of DOFs

Usage: python kinematic_chain_il.py [number_of_dof]

Author: Generated for kinematic chain imitation learning
Requirements: PySide6, torch, torchvision, numpy, opencv-python
"""

import sys
import math
import random
import argparse
import numpy as np
from typing import Tuple, List, Optional
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
    import cv2
    from collections import deque
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/OpenCV not available. Running in demo mode.")


class KinematicChainIL:
    """
    Kinematic chain with imitation learning capabilities.
    
    Generates expert demonstrations and provides supervised learning data.
    """
    
    def __init__(self, num_dof=2, arm_length=80):
        """
        Initialize the imitation learning kinematic chain.
        
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
        
        # Demonstration tracking
        self.demo_count = 0
        self.current_demo_step = 0
        self.demo_complete = False
        
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
        """Generate a new random scenario for demonstration."""
        # Random starting configuration
        self.angles = [random.uniform(-math.pi, math.pi) for _ in range(self.num_dof)]
        self.update_positions()
        
        # Generate reachable target
        self.target_pos = self.generate_reachable_target()
        
        # Reset demonstration tracking
        self.demo_count += 1
        self.current_demo_step = 0
        self.demo_complete = False
        
        print(f"Demo {self.demo_count}: New scenario generated")
        print(f"  Start EE: ({self.end_effector_pos[0]:.1f}, {self.end_effector_pos[1]:.1f})")
        print(f"  Target: ({self.target_pos[0]:.1f}, {self.target_pos[1]:.1f})")
        print(f"  Initial distance: {self.get_distance_to_target():.1f}px")
    
    def generate_reachable_target(self) -> Tuple[float, float]:
        """Generate a target that is definitely reachable."""
        if self.num_dof == 1:
            # For 1 DOF: Target MUST be exactly on the circle with radius = arm_length
            # This is the ONLY reachable workspace for 1 DOF
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
        Generate expert action using optimal inverse kinematics for 1 DOF.
        
        For 1 DOF: Calculate the shortest angular path to target.
        
        Returns:
            int: Best action (DOF_i * 3 + direction, where direction: 0=no_change, 1=-1°, 2=+1°)
        """
        if self.is_target_reached():
            return 0  # DOF 0, no change
        
        if self.num_dof == 1:
            # For 1 DOF: Use analytical solution instead of testing
            target_x, target_y = self.target_pos
            
            # Calculate target angle
            target_angle = math.atan2(target_y, target_x)
            
            # Current angle (cumulative)
            current_angle = self.angles[0]
            
            # Calculate angular difference
            angle_diff = target_angle - current_angle
            
            # Normalize angle difference to [-π, π]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Choose direction based on shortest path
            if abs(angle_diff) < math.radians(0.5):  # Very close
                return 0  # No change needed
            elif angle_diff > 0:
                return 2  # Turn right (+1°)
            else:
                return 1  # Turn left (-1°)
        
        else:
            # For multi-DOF: Use the old testing approach
            best_action = 0
            best_distance = self.get_distance_to_target()
            
            # Test all possible single-DOF movements
            for dof in range(self.num_dof):
                for direction in [1, 2]:  # Only test -1° and +1°, skip no_change
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
        Execute an action and return whether demo is complete.
        
        Action encoding: DOF_i * 3 + direction
        - direction 0: no change
        - direction 1: -1° 
        - direction 2: +1°
        
        Args:
            action (int): Action to execute
            
        Returns:
            bool: True if demonstration is complete
        """
        # Decode action: DOF_index * 3 + direction
        joint_index = action // 3
        direction = action % 3
        
        # Apply action
        if direction == 0:
            # No change for this DOF
            pass
        elif direction == 1:
            # Decrease angle by 1 degree
            self.angles[joint_index] -= self.angle_step
        else:  # direction == 2
            # Increase angle by 1 degree
            self.angles[joint_index] += self.angle_step
        
        # Keep angles in reasonable range
        if joint_index < len(self.angles):
            self.angles[joint_index] = ((self.angles[joint_index] + 2*math.pi) % (4*math.pi)) - 2*math.pi
        
        self.update_positions()
        self.current_demo_step += 1
        
        # Check if demo is complete
        if self.is_target_reached():
            self.demo_complete = True
            print(f"  Demo {self.demo_count} completed in {self.current_demo_step} steps!")
            return True
        
        # Prevent infinite demonstrations
        if self.current_demo_step > 100:
            self.demo_complete = True
            print(f"  Demo {self.demo_count} timed out after {self.current_demo_step} steps")
            return True
        
        return False
    
    def get_action_space_size(self) -> int:
        """Get total number of possible actions: 3 actions per DOF."""
        return 3 * self.num_dof


class ImitationCNN(nn.Module):
    """
    Convolutional Neural Network for imitation learning.
    
    Maps images to action probabilities.
    """
    
    def __init__(self, action_space_size: int):
        """
        Initialize the imitation learning CNN.
        
        Args:
            action_space_size (int): Number of possible actions
        """
        super(ImitationCNN, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate feature size dynamically
        self.fc_input_size = None
        
        # Fully connected layers - compact for efficiency
        self.fc1 = None  # Initialized dynamically
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space_size)  # Output layer
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def _get_conv_output_size(self, input_shape, device):
        """Calculate convolutional output size."""
        dummy_input = torch.zeros(1, *input_shape).to(device)
        with torch.no_grad():
            x = self.relu(self.conv1(dummy_input))
            print(f"After conv1: {x.shape}")
            x = self.relu(self.conv2(x))
            print(f"After conv2: {x.shape}")
            x = self.relu(self.conv3(x))
            print(f"After conv3: {x.shape}")
            flattened_size = x.view(1, -1).size(1)
            print(f"Flattened for MLP: {flattened_size} features")
            return flattened_size
    
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
        x = self.fc3(x)  # Raw logits for cross-entropy loss
        
        return x


class ImitationLearner:
    """
    Manages the imitation learning process.
    
    Collects expert demonstrations and trains the CNN.
    """
    
    def __init__(self, action_space_size: int):
        """
        Initialize the imitation learner.
        
        Args:
            action_space_size (int): Number of possible actions
        """
        self.action_space_size = action_space_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network and optimizer
        self.model = ImitationCNN(action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        
        # Data storage
        self.demo_images = deque(maxlen=5000)  # Store demonstration images
        self.demo_actions = deque(maxlen=5000)  # Store corresponding actions
        
        # Training parameters - MORE FREQUENT TRAINING
        self.batch_size = 16
        self.train_interval = 10  # Train every 10 demos instead of 50
        
        # Statistics tracking for learning progress
        self.total_demos = 0
        self.training_loss = 0.0
        self.model_accuracy = 0.0
        
        # Performance tracking for visualization
        self.loss_history = deque(maxlen=100)  # Store recent losses
        self.accuracy_history = deque(maxlen=100)  # Store recent accuracies
        self.student_performance = deque(maxlen=50)  # Store student demo lengths
        
        print(f"Imitation Learner initialized on {self.device}")
    
    def add_demonstration(self, image: np.ndarray, action: int):
        """
        Add a demonstration example to the dataset.
        
        Args:
            image (np.ndarray): State image
            action (int): Expert action
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Store demonstration
        self.demo_images.append(processed_image)
        self.demo_actions.append(action)
        self.total_demos += 1
        
        # Train more frequently for better learning
        if len(self.demo_images) >= self.batch_size and self.total_demos % self.train_interval == 0:
            self.train_model()
            
        # Print action distribution every 200 demos (less spam)
        if self.total_demos % 200 == 0:
            self.print_action_distribution()
    
    def get_learning_metrics(self):
        """Get current learning progress metrics."""
        if len(self.loss_history) == 0:
            return None
        
        # Recent performance (last 10 training iterations)
        recent_loss = sum(list(self.loss_history)[-10:]) / min(10, len(self.loss_history))
        recent_accuracy = sum(list(self.accuracy_history)[-10:]) / min(10, len(self.accuracy_history))
        
        # Student performance trend
        if len(self.student_performance) >= 5:
            recent_student_steps = sum(list(self.student_performance)[-5:]) / 5
            early_student_steps = sum(list(self.student_performance)[:5]) / 5 if len(self.student_performance) >= 10 else recent_student_steps
            improvement = early_student_steps - recent_student_steps
        else:
            recent_student_steps = 0
            improvement = 0
        
        return {
            'recent_loss': recent_loss,
            'recent_accuracy': recent_accuracy,
            'student_avg_steps': recent_student_steps,
            'student_improvement': improvement,
            'total_training_steps': len(self.loss_history)
        }
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for neural network input.
        
        Args:
            image (np.ndarray): Raw image array (H, W, C)
            
        Returns:
            torch.Tensor: Preprocessed tensor
        """
        # Resize to 84x84 (standard RL/IL image size)
        image_resized = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize and convert to CHW format
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        
        # Convert to tensor
        return torch.FloatTensor(image_chw)
    
    def train_model(self):
        """Train the model on collected demonstrations."""
        if len(self.demo_images) < self.batch_size:
            return
        
        # Sample batch
        batch_size = min(self.batch_size, len(self.demo_images))
        indices = random.sample(range(len(self.demo_images)), batch_size)
        
        batch_images = torch.stack([self.demo_images[i] for i in indices]).to(self.device)
        batch_actions = torch.LongTensor([self.demo_actions[i] for i in indices]).to(self.device)
        
        # Forward pass
        predictions = self.model(batch_images)
        loss = self.criterion(predictions, batch_actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predicted_actions = predictions.argmax(dim=1)
            accuracy = (predicted_actions == batch_actions).float().mean()
            self.model_accuracy = accuracy.item()
            self.training_loss = loss.item()
            
            # Store for visualization
            self.loss_history.append(self.training_loss)
            self.accuracy_history.append(self.model_accuracy)
        
        # Only print every 10th training iteration to reduce spam
        if len(self.loss_history) % 10 == 0:
            print(f"TRAINING UPDATE: Loss={self.training_loss:.3f}, Accuracy={self.model_accuracy:.3f}, Total Demos={self.total_demos}")
    
    def record_student_performance(self, steps_taken: int):
        """Record how many steps the student needed to complete a demo."""
        self.student_performance.append(steps_taken)
        print(f"STUDENT PERFORMANCE: Completed in {steps_taken} steps")
    
    def print_action_distribution(self):
        """Print action distribution for debugging."""
        if len(self.demo_actions) >= 100:
            # Only look at recent actions, not the entire history
            recent_count = min(100, len(self.demo_actions))
            recent_actions = list(self.demo_actions)[-recent_count:]
            action_counts = {}
            for a in recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            
            print(f"ACTION DISTRIBUTION (last {recent_count} demos): {action_counts}")
            
            # Interpret for 1 DOF case
            if len(action_counts) <= 3:
                no_action = action_counts.get(0, 0)
                left_action = action_counts.get(1, 0)  
                right_action = action_counts.get(2, 0)
                print(f"  No-action: {no_action}, Left(-1°): {left_action}, Right(+1°): {right_action}")
                
                total_moves = left_action + right_action
                if total_moves > 0:
                    left_ratio = left_action / total_moves
                    right_ratio = right_action / total_moves
                    
                    if no_action > recent_count * 0.5:
                        print("  WARNING: Too many no-action choices - expert might be stuck!")
                    elif left_ratio > 0.8 or right_ratio > 0.8:
                        print(f"  WARNING: Expert heavily biased to one direction! Left:{left_ratio:.1%}, Right:{right_ratio:.1%}")
                    else:
                        print(f"  GOOD: Balanced directions - Left:{left_ratio:.1%}, Right:{right_ratio:.1%}")
                else:
                    print("  ERROR: Expert never moves - this should not happen!")
    
    def predict_action(self, image: np.ndarray) -> int:
        """
        Predict action using trained model.
        
        Args:
            image (np.ndarray): Current state image
            
        Returns:
            int: Predicted action
        """
        if self.total_demos < self.batch_size:
            return 0  # Return no-action if not trained yet
        
        with torch.no_grad():
            processed_image = self.preprocess_image(image).unsqueeze(0).to(self.device)
            predictions = self.model(processed_image)
            return predictions.argmax().item()


class KinematicWidgetIL(QWidget):
    """Visualization widget for imitation learning."""
    
    def __init__(self, kinematic_chain):
        """
        Initialize the visualization widget.
        
        Args:
            kinematic_chain (KinematicChainIL): The kinematic chain
        """
        super().__init__()
        self.chain = kinematic_chain
        
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
            f"Demo: {self.chain.demo_count} | Step: {self.chain.current_demo_step}",
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


class MainWindowIL(QMainWindow):
    """Main window for imitation learning demonstration."""
    
    def __init__(self, num_dof=2):
        """
        Initialize the main window.
        
        Args:
            num_dof (int): Number of degrees of freedom
        """
        super().__init__()
        
        # Configure window
        self.setWindowTitle(f"Kinematic Chain Imitation Learning - {num_dof} DOF")
        self.setFixedSize(420, 600)
        
        # Create system components
        self.chain = KinematicChainIL(num_dof)
        
        if TORCH_AVAILABLE:
            self.learner = ImitationLearner(self.chain.get_action_space_size())
            self.learning_enabled = True
        else:
            self.learner = None
            self.learning_enabled = False
        
        # Set up UI
        self._setup_ui()
        
        # Demo control
        self.demo_mode = "expert"  # "expert" or "student"
        self.steps_per_second = 0
        self.last_time = time.time()
        self.step_counter = 0
        
        # Set up timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.timer.start(20)  # 50 FPS
        
        print(f"Imitation Learning started with {num_dof} DOF")
    
    def _setup_ui(self):
        """Set up user interface."""
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Kinematic visualization
        self.kinematic_widget = KinematicWidgetIL(self.chain)
        layout.addWidget(self.kinematic_widget)
        
        # Status information
        info_layout = QVBoxLayout()
        
        self.mode_label = QLabel("Mode: Expert Demonstration")
        self.stats_label = QLabel("Statistics: Collecting data...")
        self.performance_label = QLabel("Performance: Starting up...")
        
        info_layout.addWidget(self.mode_label)
        info_layout.addWidget(self.stats_label)
        info_layout.addWidget(self.performance_label)
        
        layout.addLayout(info_layout)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def step(self):
        """Perform one step of the demonstration/learning process."""
        # Performance monitoring
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.steps_per_second = self.step_counter / (current_time - self.last_time)
            self.step_counter = 0
            self.last_time = current_time
        
        self.step_counter += 1
        
        if not self.learning_enabled:
            # Demo mode without learning
            self._expert_step()
            self.kinematic_widget.update()
            return
        
        # Capture current image
        current_image = self.kinematic_widget.capture_image()
        
        if self.demo_mode == "expert":
            # Expert demonstration phase
            expert_action = self.chain.get_expert_action()
            
            # Store demonstration
            self.learner.add_demonstration(current_image, expert_action)
            
            # Execute expert action
            done = self.chain.take_action(expert_action)
            
            if done:
                # Switch to student mode every 5 demos for more frequent testing
                if self.chain.demo_count % 5 == 0 and self.learner.total_demos > 50:
                    self.demo_mode = "student"
                    print("Switching to student mode...")
                else:
                    self.chain.generate_new_scenario()
        
        else:  # student mode
            # Let student try
            student_action = self.learner.predict_action(current_image)
            done = self.chain.take_action(student_action)
            
            if done:
                # Record student performance
                self.learner.record_student_performance(self.chain.current_demo_step)
                # Switch back to expert mode
                self.demo_mode = "expert"
                self.chain.generate_new_scenario()
                print("Switching back to expert mode...")
        
        # Update UI
        self._update_status()
        self.kinematic_widget.update()
    
    def _expert_step(self):
        """Perform expert demonstration without learning."""
        action = self.chain.get_expert_action()
        done = self.chain.take_action(action)
        
        if done:
            self.chain.generate_new_scenario()
    
    def _update_status(self):
        """Update status labels with clearer learning progress indicators."""
        mode_text = f"Mode: {'Expert' if self.demo_mode == 'expert' else 'Student'} | Speed: {self.steps_per_second:.0f} steps/sec"
        self.mode_label.setText(mode_text)
        
        if self.learning_enabled:
            # Get learning metrics
            metrics = self.learner.get_learning_metrics()
            
            if metrics and metrics['total_training_steps'] > 0:
                # Learning progress assessment
                accuracy = metrics['recent_accuracy']
                if accuracy > 0.8:
                    learning_status = "✅ EXCELLENT LEARNING"
                elif accuracy > 0.7:
                    learning_status = "✓ GOOD LEARNING" 
                elif accuracy > 0.6:
                    learning_status = "~ MODERATE LEARNING"
                elif accuracy > 0.5:
                    learning_status = "⚠ SLOW LEARNING"
                else:
                    learning_status = "❌ POOR LEARNING"
                
                stats_text = f"Demos: {self.learner.total_demos} | Loss: {metrics['recent_loss']:.3f} | Accuracy: {accuracy:.3f} | {learning_status}"
                
                # Student performance analysis
                if metrics['student_avg_steps'] > 0:
                    if metrics['student_improvement'] > 5:
                        student_status = f"Student: {metrics['student_avg_steps']:.1f} steps (IMPROVED ↑{metrics['student_improvement']:.1f})"
                    elif metrics['student_improvement'] < -5:
                        student_status = f"Student: {metrics['student_avg_steps']:.1f} steps (WORSE ↓{abs(metrics['student_improvement']):.1f})"
                    else:
                        student_status = f"Student: {metrics['student_avg_steps']:.1f} steps (stable)"
                else:
                    student_status = "Student: Not tested yet"
                
                perf_text = f"{student_status} | Distance: {self.chain.get_distance_to_target():.1f}px"
                
            else:
                stats_text = f"Demos: {self.learner.total_demos} | Collecting initial training data..."
                perf_text = f"Distance: {self.chain.get_distance_to_target():.1f}px | Warming up neural network..."
        else:
            stats_text = f"Demo mode - no learning | Distance: {self.chain.get_distance_to_target():.1f}px"
            perf_text = "Install PyTorch for imitation learning"
        
        self.stats_label.setText(stats_text)
        self.performance_label.setText(perf_text)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Imitation Learning for Kinematic Chain Control",
        epilog="Example: python kinematic_chain_il.py 2"
    )
    parser.add_argument(
        "num_dof", 
        nargs="?", 
        type=int, 
        default=2,
        help="Number of degrees of freedom. Default: 2"
    )
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    if args.num_dof < 1:
        print("Error: Number of DOF must be at least 1.")
        return 1
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindowIL(args.num_dof)
    window.show()
    
    # Print startup information
    print(f"\nKinematic Chain Imitation Learning")
    print(f"  • Degrees of Freedom: {args.num_dof}")
    print(f"  • Action Space Size: {3 * args.num_dof}")
    print(f"  • PyTorch Available: {TORCH_AVAILABLE}")
    print(f"  • Learning: {'Enabled' if TORCH_AVAILABLE else 'Demo Only'}")
    print(f"\nExpert Strategy:")
    print(f"  • Tests all single-DOF movements")
    print(f"  • Picks action that reduces distance most")
    print(f"  • Moves only one joint at a time")
    print(f"\nProcess:")
    print(f"  • Expert demonstrates → Student learns → Test student → Repeat")
    print(f"  • Switches to student mode every 10 demos")
    
    if not TORCH_AVAILABLE:
        print(f"\nInstall dependencies: pip install torch opencv-python")
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())