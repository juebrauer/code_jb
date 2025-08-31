#!/usr/bin/env python3
"""
Deep Reinforcement Learning for Kinematic Chain Control

This script combines a kinematic chain simulation with Deep RL to learn
inverse kinematics. The agent learns to reach target positions using only
visual input (rendered images) and discrete angle adjustments.

Features:
- Image-based RL environment (agent sees only the rendered scene)
- Discrete action space: +1° or -1° for each DOF
- Random target generation within reachable workspace
- Episode-based training with automatic reset on target reach
- Configurable number of DOFs (starting with 1 DOF for simplicity)

Usage: python kinematic_chain_rl.py [number_of_dof]

Author: Generated for kinematic chain RL visualization  
Requirements: PySide6, torch, torchvision, numpy
"""

import sys
import math
import random
import argparse
import numpy as np
from typing import Tuple, List, Optional

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
    from collections import deque
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Running in demo mode without RL training.")


class KinematicChainRL:
    """
    Enhanced kinematic chain with reinforcement learning capabilities.
    
    This class extends the basic kinematic chain to include RL-specific
    functionality like reward calculation, episode management, and
    action space definition.
    """
    
    def __init__(self, num_dof=1, arm_length=80):
        """
        Initialize the RL-enabled kinematic chain.
        
        Args:
            num_dof (int): Number of degrees of freedom (default: 1 for initial experiments)
            arm_length (float): Length of each arm segment in pixels (default: 80)
        """
        self.num_dof = num_dof
        self.arm_length = arm_length
        self.max_reach = num_dof * arm_length  # Maximum reachable distance
        
        # Initialize joint angles (start at zero for consistent training)
        self.angles = [0.0 for _ in range(num_dof)]
        
        # Position tracking
        self.joint_positions = [(0, 0)]
        self.end_effector_pos = (0, 0)
        
        # RL-specific parameters
        self.target_pos = (0, 0)
        self.target_tolerance = 15.0  # Pixels - target is reached within this radius
        self.angle_step = math.radians(1)  # 1 degree steps
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.max_steps_per_episode = 500  # Prevent infinite episodes
        
        # Performance metrics
        self.success_count = 0
        self.total_episodes = 0
        
        # Initialize positions and target
        self.update_positions()
        self.reset_episode()
    
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
    
    def generate_random_target(self) -> Tuple[float, float]:
        """
        Generate a random target position within the DEFINITELY reachable workspace.
        
        For 1 DOF: Target must be on a circle with radius = arm_length
        For multiple DOFs: More complex workspace, but we ensure reachability
        
        Returns:
            Tuple[float, float]: Target position (x, y)
        """
        if self.num_dof == 1:
            # For 1 DOF: Target MUST be exactly on the circle with radius = arm_length
            # This ensures the target is always reachable
            angle = random.uniform(0, 2 * math.pi)
            target_x = self.arm_length * math.cos(angle)
            target_y = self.arm_length * math.sin(angle)
            return (target_x, target_y)
        else:
            # For multiple DOFs: Use conservative estimate within workspace
            max_distance = 0.8 * self.max_reach  # More conservative
            min_distance = 0.2 * self.max_reach  # Avoid too close to origin
            
            distance = random.uniform(min_distance, max_distance)
            angle = random.uniform(0, 2 * math.pi)
            
            target_x = distance * math.cos(angle)
            target_y = distance * math.sin(angle)
            return (target_x, target_y)
    
    def reset_episode(self):
        """Reset the environment for a new episode."""
        # Reset joint angles to random starting positions
        self.angles = [random.uniform(-math.pi, math.pi) for _ in range(self.num_dof)]
        
        # Generate new target
        self.target_pos = self.generate_random_target()
        
        # Reset counters
        self.step_count = 0
        self.episode_count += 1
        
        # Update positions
        self.update_positions()
        
        print(f"Episode {self.episode_count}: New target at ({self.target_pos[0]:.1f}, {self.target_pos[1]:.1f})")
    
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
    
    def get_reward(self) -> float:
        """
        Calculate reward for the current state.
        
        Reward structure:
        - Large positive reward for reaching target
        - Negative reward proportional to distance (encourages getting closer)
        - Small penalty per step (encourages efficiency)
        - Large penalty for exceeding max steps
        
        Returns:
            float: Reward value
        """
        distance = self.get_distance_to_target()
        
        if self.is_target_reached():
            return 100.0  # Large positive reward for success
        
        # Distance-based reward (negative, closer to target is better)
        distance_reward = -distance / 10.0
        
        # Step penalty (encourages efficiency)
        step_penalty = -0.1
        
        # Timeout penalty
        if self.step_count >= self.max_steps_per_episode:
            return -50.0  # Large negative reward for timeout
        
        return distance_reward + step_penalty
    
    def take_action(self, action: int) -> Tuple[float, bool]:
        """
        Execute an action and return reward and episode termination status.
        
        Action space:
        - For 1 DOF: 0 = -1°, 1 = +1°
        - For n DOFs: 2*i = DOF i -1°, 2*i+1 = DOF i +1°
        
        Args:
            action (int): Action index
            
        Returns:
            Tuple[float, bool]: (reward, episode_done)
        """
        if action < 0 or action >= 2 * self.num_dof:
            raise ValueError(f"Invalid action {action} for {self.num_dof} DOFs")
        
        # Decode action
        joint_index = action // 2
        direction = 1 if action % 2 == 1 else -1
        
        # Apply action
        self.angles[joint_index] += direction * self.angle_step
        
        # Keep angles in reasonable range (-2π to 2π)
        self.angles[joint_index] = ((self.angles[joint_index] + 2*math.pi) % (4*math.pi)) - 2*math.pi
        
        # Update positions
        self.update_positions()
        self.step_count += 1
        
        # Calculate reward and check termination
        reward = self.get_reward()
        episode_done = self.is_target_reached() or self.step_count >= self.max_steps_per_episode
        
        if episode_done and self.is_target_reached():
            self.success_count += 1
            print(f"🎯 Target reached in {self.step_count} steps! Success rate: {self.success_count}/{self.episode_count}")
        elif episode_done:
            print(f"⏰ Episode timeout after {self.step_count} steps")
        
        return reward, episode_done
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space."""
        return 2 * self.num_dof


class DQN(nn.Module):
    """
    Deep Q-Network for learning kinematic control from images.
    
    Architecture:
    - Convolutional layers for image processing
    - Fully connected layers for decision making
    - Output layer with one value per action
    """
    
    def __init__(self, action_space_size: int):
        """
        Initialize the DQN.
        
        Args:
            action_space_size (int): Number of possible actions
        """
        super(DQN, self).__init__()
        
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # We'll calculate the actual size dynamically
        self.fc_input_size = None
        
        # Fully connected layers (will be initialized after first forward pass)
        self.fc1 = None
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_space_size)
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def _get_conv_output_size(self, input_shape, device):
        """Calculate the output size of convolutional layers."""
        # Create a dummy input to calculate the output size on correct device
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
            device = x.device  # Get device from input tensor
            self.fc_input_size = self._get_conv_output_size(x.shape[1:], device)
            self.fc1 = nn.Linear(self.fc_input_size, 512).to(device)
            print(f"Initialized FC layer with input size: {self.fc_input_size}")
        
        # Convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class RLAgent:
    """
    Deep Q-Learning agent for kinematic control.
    
    Implements DQN with experience replay and target network.
    """
    
    def __init__(self, action_space_size: int, learning_rate: float = 1e-4):
        """
        Initialize the RL agent.
        
        Args:
            action_space_size (int): Number of possible actions
            learning_rate (float): Learning rate for optimization
        """
        self.action_space_size = action_space_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(action_space_size).to(self.device)
        self.target_network = DQN(action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay - REDUCED buffer size for memory efficiency
        self.memory = deque(maxlen=2000)  # Reduced from 10000
        self.batch_size = 8  # Reduced from 32 for memory efficiency
        
        # RL parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
        self.target_update_frequency = 100  # Steps between target network updates
        
        # Training tracking
        self.step_count = 0
        
        # Update target network initially
        self.update_target_network()
        
        print(f"RL Agent initialized on {self.device}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for neural network input with memory optimization.
        
        Args:
            image (np.ndarray): Raw image array (H, W, C)
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for network
        """
        # Resize image to reduce memory footprint while preserving aspect ratio
        # For 400x400 -> 84x84 (common RL image size)
        import cv2
        image_resized = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Convert to float and normalize to [0, 1]
        image_resized = image_resized.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        image_processed = np.transpose(image_resized, (2, 0, 1))
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(image_processed).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def select_action(self, state: torch.Tensor) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Current state (preprocessed image)
            
        Returns:
            int: Selected action
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_space_size - 1)
        else:
            # Exploitation: best action according to Q-network
            with torch.no_grad():
                q_values = self.q_network(state)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the Q-network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        try:
            # Convert to tensors with gradient checkpointing for memory efficiency
            states = torch.cat(states)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.cat(next_states)
            dones = torch.BoolTensor(dones).to(self.device)
            
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute loss and optimize
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Clear intermediate tensors to free GPU memory
            del states, next_states, current_q_values, next_q_values, target_q_values
            torch.cuda.empty_cache()  # Force GPU memory cleanup
            
        except torch.cuda.OutOfMemoryError:
            print("⚠️  GPU OOM during training - clearing cache and skipping batch")
            torch.cuda.empty_cache()
            return
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


class KinematicWidgetRL(QWidget):
    """
    Enhanced visualization widget with RL-specific features.
    
    Adds target visualization and image capture for RL training.
    """
    
    def __init__(self, kinematic_chain):
        """
        Initialize the RL visualization widget.
        
        Args:
            kinematic_chain (KinematicChainRL): The RL-enabled kinematic chain
        """
        super().__init__()
        self.chain = kinematic_chain
        
        # Set fixed window size - REDUCED for memory efficiency
        self.setFixedSize(400, 400)  # Reduced from 800x800
        
        # Calculate center coordinates
        self.center_x = 200  # Updated for new size
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


class MainWindowRL(QMainWindow):
    """
    Main window for RL-enabled kinematic chain simulation.
    
    Manages the RL training loop and visualization updates.
    """
    
    def __init__(self, num_dof=1):
        """
        Initialize the RL main window.
        
        Args:
            num_dof (int): Number of degrees of freedom
        """
        super().__init__()
        
        # Configure window - REDUCED size for memory efficiency
        self.setWindowTitle(f"Kinematic Chain Deep RL - {num_dof} DOF")
        self.setFixedSize(420, 550)  # Smaller window for 400x400 widget
        
        # Create kinematic chain and agent
        self.chain = KinematicChainRL(num_dof)
        
        if TORCH_AVAILABLE:
            self.agent = RLAgent(self.chain.get_action_space_size())
            self.training_enabled = True
        else:
            self.agent = None
            self.training_enabled = False
        
        # Set up UI
        self._setup_ui()
        
        # RL training parameters
        self.training_speed = 1  # milliseconds between steps - MINIMAL for max speed
        self.current_state = None
        
        # Performance monitoring
        self.steps_per_second = 0
        self.last_time = 0
        self.step_counter = 0
        
        # Set up training timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.rl_step)
        self.timer.start(self.training_speed)
        
        print(f"🚀 RL Training Started (Training: {self.training_enabled})")
    
    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Kinematic visualization
        self.kinematic_widget = KinematicWidgetRL(self.chain)
        layout.addWidget(self.kinematic_widget)
        
        # RL status information
        info_layout = QHBoxLayout()
        
        self.epsilon_label = QLabel("Epsilon: N/A")
        self.training_status_label = QLabel("Training: Disabled" if not self.training_enabled else "Training: Enabled")
        
        info_layout.addWidget(self.epsilon_label)
        info_layout.addWidget(self.training_status_label)
        
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