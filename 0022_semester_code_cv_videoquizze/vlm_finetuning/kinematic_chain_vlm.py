#!/usr/bin/env python3
"""
Kinematic Chain Control with VLM (Paligemma) Fine-tuning

This script provides three modes:
1. collect: Generate and save training data using expert demonstrations (VLM format)
2. train: Fine-tune Paligemma VLM on collected data
3. test: Load trained VLM and use it to control the kinematic chain

Usage:
    python kinematic_chain_vlm.py collect --dof 2 --samples 5000 --output data/dof2_vlm
    python kinematic_chain_vlm.py train --dof 2 --data data/dof2_vlm --epochs 5 --output models/
    python kinematic_chain_vlm.py test --dof 2 --model models/paligemma_dof2

Author: VLM-based Imitation Learning for Kinematic Chains
Requirements: PySide6, transformers, peft, bitsandbytes, accelerate, torch, numpy, opencv-python, pandas
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
from pathlib import Path

# Qt imports
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap
from PySide6.QtCore import Qt

# Deep Learning imports
try:
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig
    import cv2
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"Warning: Required packages not available: {e}")


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
        Apply an action and update the chain state.
        
        Args:
            action (int): Action to apply
        
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
        
        # Continue until target is reached (no timeout)
        return False
    
    def get_action_space_size(self) -> int:
        """Get total number of possible actions."""
        return 3 * self.num_dof
    
    def action_to_text(self, action: int) -> str:
        """
        Convert action index to natural language description.
        
        Args:
            action (int): Action index
        
        Returns:
            str: Natural language description of the action
        """
        joint_idx = action // 3
        direction = action % 3
        
        if direction == 0:
            return "No movement needed - target reached or no action required"
        elif direction == 1:
            return f"Rotate joint {joint_idx} counter-clockwise by 1 degree"
        else:  # direction == 2
            return f"Rotate joint {joint_idx} clockwise by 1 degree"
    
    def action_to_structured(self, action: int) -> dict:
        """
        Convert action index to structured format.
        
        Args:
            action (int): Action index
        
        Returns:
            dict: Structured action {joint, direction}
        """
        joint_idx = action // 3
        direction = action % 3
        
        # Map direction: 0 -> 0, 1 -> -1, 2 -> 1
        direction_value = 0 if direction == 0 else (-1 if direction == 1 else 1)
        
        return {
            "joint": joint_idx,
            "direction": direction_value
        }


class VLMDataset(Dataset):
    """Dataset for Paligemma VLM training."""
    
    def __init__(self, data_dir: str, processor):
        """
        Initialize dataset.
        
        Args:
            data_dir (str): Directory containing training data
            processor: Paligemma processor
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        
        # Load CSV data
        csv_path = self.data_dir / "training_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.num_samples = len(self.df)
        print(f"Loaded VLM dataset with {self.num_samples} samples from CSV")
        print(f"  CSV columns: {list(self.df.columns)}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        row = self.df.iloc[idx]
        
        # Load image
        image_path = self.data_dir / row['image_filename']
        image = Image.open(image_path).convert('RGB')
        
        # Get prompt and answer
        prompt = row['prompt']
        answer = row['answer']
        
        # For Paligemma, we just need the text WITHOUT manually adding image tokens
        # The processor will handle image token placement automatically
        # Format: prompt + "\n" + answer for training
        full_text = f"{prompt}\n{answer}"
        
        # Process with Paligemma processor
        # The processor will automatically add the correct number of image tokens
        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            # Don't use padding/truncation here - let the model handle it
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs


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
        
        # Set fixed size for consistent visualization
        self.setFixedSize(400, 400)
        
        # Center point
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
        brush = QBrush(QColor(0, 0, 255))  # Blue circle
        painter.setBrush(brush)
        painter.setPen(QPen(QColor(0, 0, 255), 1))
        
        ee_x = self.center_x + self.chain.end_effector_pos[0]
        ee_y = self.center_y + self.chain.end_effector_pos[1]
        
        painter.drawEllipse(
            int(ee_x - self.end_effector_radius), 
            int(ee_y - self.end_effector_radius),
            2 * self.end_effector_radius, 
            2 * self.end_effector_radius
        )
    
    def _draw_target(self, painter):
        """Draw target as red X marker."""
        target_x = self.center_x + self.chain.target_pos[0]
        target_y = self.center_y + self.chain.target_pos[1]
        
        # Draw red X
        pen = QPen(QColor(255, 0, 0), 3)
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
    
    def _draw_info_text(self, painter):
        """Draw information text."""
        painter.setPen(QColor(0, 0, 0))
        distance = self.chain.get_distance_to_target()
        info_text = f"Distance: {distance:.1f}px | Step: {self.chain.current_step}"
        painter.drawText(10, 20, info_text)
    
    def capture_image(self) -> np.ndarray:
        """
        Capture the current visualization as a numpy array.
        
        Returns:
            np.ndarray: RGB image array
        """
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        
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
        self.setWindowTitle(f"Kinematic Chain VLM - {mode.upper()} Mode - {chain.num_dof} DOF")
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
            self.target_samples = None
        
        # Test statistics (for test mode)
        if mode == 'test':
            self.test_stats = {
                'scenarios_completed': 0,
                'total_steps': 0,
                'avg_steps_per_scenario': 0
            }
        
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
        self.mode_label = QLabel(f"Mode: {self.mode.upper()}")
        self.stats_label = QLabel("Statistics: Starting...")
        self.performance_label = QLabel("Performance: Initializing...")
        
        layout.addWidget(self.mode_label)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.performance_label)
        
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
            
            # Check if we have collected enough samples
            if self.target_samples and self.collected_samples >= self.target_samples:
                self.timer.stop()
                self._finalize_collection()
                return
        
        # Execute action
        done = self.chain.take_action(action)
        
        if done:
            if self.mode == 'test':
                self.test_stats['scenarios_completed'] += 1
                self.test_stats['avg_steps_per_scenario'] = (
                    self.test_stats['total_steps'] / self.test_stats['scenarios_completed']
                )
            self.chain.generate_new_scenario()
        
        if self.mode == 'test':
            self.test_stats['total_steps'] += 1
        
        # Update UI
        self._update_status()
        self.kinematic_widget.update()
    
    def _save_sample(self, image: np.ndarray, action: int):
        """Save a training sample to disk in VLM format."""
        # Save image
        image_path = self.data_dir / f"image_{self.collected_samples:06d}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Append to CSV data (will be saved at the end)
        if not hasattr(self, 'csv_data'):
            self.csv_data = []
        
        # Create prompt and answer for VLM
        prompt = "Analyze the robot arm position (black joints, blue end effector) and target (red X). Which joint should move and in which direction to reach the target?"
        answer = self.chain.action_to_text(action)
        
        self.csv_data.append({
            'sample_id': self.collected_samples,
            'image_filename': f"image_{self.collected_samples:06d}.png",
            'action': action,
            'scenario': self.chain.scenario_count,
            'step_in_scenario': self.chain.current_step,
            'prompt': prompt,
            'answer': answer
        })
        
        self.collected_samples += 1
    
    def _finalize_collection(self):
        """Finalize the collection process and save metadata."""
        print(f"\nCollection complete! Saved exactly {self.collected_samples} samples.")
        
        # Save CSV data
        if hasattr(self, 'csv_data') and len(self.csv_data) > 0:
            df = pd.DataFrame(self.csv_data)
            csv_path = self.data_dir / "training_data.csv"
            df.to_csv(csv_path, index=False)
            print(f"VLM CSV data saved to {csv_path}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Total rows: {len(df)}")
            
            # Show sample entries
            print(f"\nSample entry:")
            print(f"  Prompt: {df.iloc[0]['prompt']}")
            print(f"  Answer: {df.iloc[0]['answer']}")
        
        # Save metadata
        metadata = {
            'dof': self.chain.num_dof,
            'total_samples': self.collected_samples,
            'action_space_size': self.chain.get_action_space_size(),
            'arm_length': self.chain.arm_length,
            'target_tolerance': self.chain.target_tolerance,
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_format': 'VLM_Paligemma',
            'image_size': '400x400'
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
                stats_text = f"Samples: {self.collected_samples}/{self.target_samples} | Scenarios: {self.chain.scenario_count}"
            else:
                stats_text = f"Samples: {self.collected_samples} | Scenarios: {self.chain.scenario_count}"
            perf_text = f"Distance: {self.chain.get_distance_to_target():.1f}px | Step: {self.chain.current_step}"
        else:  # test mode
            stats_text = f"Scenarios: {self.test_stats['scenarios_completed']} | Avg steps: {self.test_stats['avg_steps_per_scenario']:.1f}"
            perf_text = f"Distance: {self.chain.get_distance_to_target():.1f}px | Total steps: {self.test_stats['total_steps']}"
        
        self.stats_label.setText(stats_text)
        self.performance_label.setText(perf_text)


def collect_mode(args):
    """Run collect mode to gather VLM training data."""
    print(f"\n=== COLLECT MODE (VLM Format) ===")
    print(f"DOF: {args.dof}")
    print(f"Target samples: {args.samples}")
    print(f"Output directory: {args.output}")
    
    # Create kinematic chain
    chain = KinematicChain(args.dof)
    
    # Create expert action generator that ignores the image argument
    def expert_action_generator(image):
        """Wrapper for expert action that ignores the image."""
        return chain.get_expert_action()
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow(chain, 'collect', expert_action_generator)
    window.set_data_directory(args.output)
    window.target_samples = args.samples
    window.show()
    
    print(f"\nCollecting {args.samples} samples with expert demonstrations...")
    print(f"Data will be saved in Paligemma VLM format (image + prompt + answer)")
    
    return app.exec()


def train_mode(args):
    """Run train mode to fine-tune Paligemma VLM."""
    if not TORCH_AVAILABLE:
        print("Error: Required packages not available. Please install: transformers, peft, bitsandbytes, accelerate")
        return 1
    
    print(f"\n=== TRAIN MODE (Paligemma VLM) ===")
    print(f"DOF: {args.dof}")
    print(f"Data directory: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: {args.base_model}")
    
    # Check metadata
    metadata_path = Path(args.data) / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if metadata['dof'] != args.dof:
        print(f"Error: DOF mismatch. Data has {metadata['dof']} DOF, but training with {args.dof} DOF.")
        return 1
    
    print(f"Action space size: {metadata['action_space_size']}")
    print(f"Total samples: {metadata['total_samples']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load Paligemma model and processor
    print(f"\nLoading Paligemma model: {args.base_model}")
    
    # Quantization config for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    try:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.base_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have:")
        print("1. Accepted the Paligemma license on Hugging Face")
        print("2. Logged in with: huggingface-cli login")
        return 1
    
    # Configure LoRA
    print("Configuring LoRA for parameter-efficient fine-tuning...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("\nLoading VLM dataset...")
    try:
        dataset = VLMDataset(args.data, processor)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Custom collate function for batching with padding
    def collate_fn(batch):
        """Collate function to pad batches properly."""
        # Find max length in batch
        max_length = max(item['input_ids'].shape[0] for item in batch)
        
        # Pad each item to max_length
        padded_batch = {
            'input_ids': [],
            'attention_mask': [],
            'pixel_values': []
        }
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            
            # Calculate padding needed
            padding_length = max_length - input_ids.shape[0]
            
            if padding_length > 0:
                # Pad input_ids with pad_token_id (typically 0)
                input_ids = torch.cat([
                    input_ids,
                    torch.zeros(padding_length, dtype=input_ids.dtype)
                ])
                # Pad attention_mask with 0s
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=attention_mask.dtype)
                ])
            
            padded_batch['input_ids'].append(input_ids)
            padded_batch['attention_mask'].append(attention_mask)
            padded_batch['pixel_values'].append(item['pixel_values'])
        
        # Stack into tensors
        padded_batch['input_ids'] = torch.stack(padded_batch['input_ids'])
        padded_batch['attention_mask'] = torch.stack(padded_batch['attention_mask'])
        padded_batch['pixel_values'] = torch.stack(padded_batch['pixel_values'])
        
        return padded_batch
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = Path(args.output) / f"paligemma_dof{args.dof}_best"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            
            print(f"  ✓ Best model saved to {output_dir}")
    
    # Save final model
    final_output_dir = Path(args.output) / f"paligemma_dof{args.dof}_final"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(final_output_dir)
    processor.save_pretrained(final_output_dir)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final model saved to: {final_output_dir}")
    print(f"Best model saved to: {Path(args.output) / f'paligemma_dof{args.dof}_best'}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return 0


def test_mode(args):
    """Run test mode with trained Paligemma VLM."""
    if not TORCH_AVAILABLE:
        print("Error: Required packages not available.")
        return 1
    
    print(f"\n=== TEST MODE (Paligemma VLM) ===")
    print(f"DOF: {args.dof}")
    print(f"Model directory: {args.model}")
    
    # Load model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    try:
        print("Loading Paligemma model...")
        
        # Load base model first
        base_model_name = args.base_model
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load LoRA weights
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model)
        
        processor = AutoProcessor.from_pretrained(args.model)
        model.eval()
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Create kinematic chain
    chain = KinematicChain(args.dof)
    
    # Create VLM action generator
    def vlm_action_generator(image: np.ndarray) -> int:
        """Generate action using VLM."""
        # Convert numpy image to PIL
        pil_image = Image.fromarray(image)
        
        # Create prompt (processor will add image tokens automatically)
        prompt = "Analyze the robot arm position (black joints, blue end effector) and target (red X). Which joint should move and in which direction to reach the target?"
        
        # Process with VLM - processor handles image tokens automatically
        inputs = processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt"
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        # Decode response
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # Extract action from text
        # The model should output something like "Rotate joint 0 clockwise by 1 degree"
        action = parse_action_from_text(generated_text, chain.num_dof)
        
        if args.verbose:
            print(f"VLM Response: {generated_text}")
            print(f"Parsed Action: {action}")
        
        return action
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow(chain, 'test', vlm_action_generator)
    window.show()
    
    print(f"\nTest mode running. Paligemma VLM is controlling the arm.")
    return app.exec()


def parse_action_from_text(text: str, num_dof: int) -> int:
    """
    Parse action from VLM generated text.
    
    Args:
        text (str): Generated text from VLM
        num_dof (int): Number of degrees of freedom
    
    Returns:
        int: Parsed action index
    """
    text_lower = text.lower()
    
    # Check for "no movement"
    if "no movement" in text_lower or "target reached" in text_lower:
        return 0
    
    # Extract joint number
    joint_idx = 0
    for i in range(num_dof):
        if f"joint {i}" in text_lower:
            joint_idx = i
            break
    
    # Extract direction
    if "counter-clockwise" in text_lower or "ccw" in text_lower:
        direction = 1  # -1 degree
    elif "clockwise" in text_lower or "cw" in text_lower:
        direction = 2  # +1 degree
    else:
        direction = 0  # No change
    
    action = joint_idx * 3 + direction
    return action


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kinematic Chain Control with Paligemma VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 5000 training samples with 2 DOF
  python kinematic_chain_vlm.py collect --dof 2 --samples 5000 --output data/dof2_vlm

  # Train Paligemma VLM on collected data
  python kinematic_chain_vlm.py train --dof 2 --data data/dof2_vlm --epochs 5 --output models/

  # Test trained VLM
  python kinematic_chain_vlm.py test --dof 2 --model models/paligemma_dof2_best
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')
    
    # Collect mode
    collect_parser = subparsers.add_parser('collect', help='Collect VLM training data')
    collect_parser.add_argument('--dof', type=int, default=2, help='Number of degrees of freedom (default: 2)')
    collect_parser.add_argument('--samples', type=int, default=5000, help='Number of samples to collect (default: 5000)')
    collect_parser.add_argument('--output', type=str, default='data/vlm_data', help='Output directory (default: data/vlm_data)')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train Paligemma VLM')
    train_parser.add_argument('--dof', type=int, default=2, help='Number of degrees of freedom (default: 2)')
    train_parser.add_argument('--data', type=str, required=True, help='Training data directory')
    train_parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    train_parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
    train_parser.add_argument('--output', type=str, default='models/', help='Output directory for saved model (default: models/)')
    train_parser.add_argument('--base-model', type=str, default='google/paligemma-3b-pt-224', help='Base Paligemma model (default: google/paligemma-3b-pt-224)')
    train_parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank (default: 8)')
    train_parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha (default: 16)')
    
    # Test mode
    test_parser = subparsers.add_parser('test', help='Test trained VLM')
    test_parser.add_argument('--dof', type=int, default=2, help='Number of degrees of freedom (default: 2)')
    test_parser.add_argument('--model', type=str, required=True, help='Path to trained model directory')
    test_parser.add_argument('--base-model', type=str, default='google/paligemma-3b-pt-224', help='Base Paligemma model (default: google/paligemma-3b-pt-224)')
    test_parser.add_argument('--verbose', action='store_true', help='Print VLM responses')
    
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
    print(f"\nKinematic Chain VLM Control - {args.mode.upper()} Mode")
    print(f"Degrees of Freedom: {args.dof}")
    print(f"PyTorch/Transformers Available: {TORCH_AVAILABLE}")
    
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