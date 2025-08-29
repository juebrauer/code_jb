#!/usr/bin/env python3
"""
Kinematic Actuator Chain Simulation with Qt Visualization

This script simulates a kinematic chain with multiple revolute degrees of freedom (DOF).
Each joint can rotate independently, and the position of the end effector is calculated
using forward kinematics. The simulation includes an animated demo that sequentially
rotates each joint through a full 360-degree revolution.

Usage: python kinematic_chain.py [number_of_dof]

Author: Generated for kinematic chain visualization
Requirements: PySide6
"""

import sys
import math
import random
import argparse
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
from PySide6.QtCore import Qt


class KinematicChain:
    """
    Represents a kinematic chain with multiple revolute degrees of freedom.
    
    This class handles the mathematical representation and calculation of a serial
    kinematic chain where each joint is a revolute joint. The chain starts from
    the origin and each subsequent link is connected to the previous one.
    
    Attributes:
        num_dof (int): Number of degrees of freedom (revolute joints)
        arm_length (float): Length of each arm segment in pixels
        angles (list): Current angles of all joints in radians
        joint_positions (list): Cartesian positions of all joints [(x, y), ...]
        end_effector_pos (tuple): Position of the end effector (x, y)
    """
    
    def __init__(self, num_dof=3, arm_length=80):
        """
        Initialize the kinematic chain.
        
        Args:
            num_dof (int): Number of degrees of freedom (default: 3)
            arm_length (float): Length of each arm segment in pixels (default: 80)
        """
        self.num_dof = num_dof
        self.arm_length = arm_length
        
        # Initialize all joint angles with random values between 0 and 2π
        self.angles = [random.uniform(0, 2 * math.pi) for _ in range(num_dof)]
        
        # Initialize position arrays
        self.joint_positions = [(0, 0)]  # Start at origin
        self.end_effector_pos = (0, 0)
        
        # Calculate initial positions
        self.update_positions()
    
    def update_positions(self):
        """
        Calculate all joint positions using forward kinematics.
        
        This method computes the Cartesian coordinates of each joint and the
        end effector based on the current joint angles. Uses cumulative angle
        transformation typical for serial manipulators.
        """
        # Reset joint positions starting from origin
        self.joint_positions = [(0, 0)]
        
        # Initialize position and cumulative angle
        x, y = 0, 0
        cumulative_angle = 0
        
        # Calculate position of each joint
        for i in range(self.num_dof):
            # Add current joint angle to cumulative rotation
            cumulative_angle += self.angles[i]
            
            # Calculate next joint position using trigonometry
            x += self.arm_length * math.cos(cumulative_angle)
            y += self.arm_length * math.sin(cumulative_angle)
            
            # Store joint position
            self.joint_positions.append((x, y))
        
        # End effector is at the last joint position
        self.end_effector_pos = (x, y)
    
    def set_angle(self, joint_index, angle):
        """
        Set the angle of a specific joint and update all positions.
        
        Args:
            joint_index (int): Index of the joint to modify (0-based)
            angle (float): New angle in radians
        """
        if 0 <= joint_index < self.num_dof:
            self.angles[joint_index] = angle
            self.update_positions()
    
    def get_angle_degrees(self, joint_index):
        """
        Get the angle of a specific joint in degrees.
        
        Args:
            joint_index (int): Index of the joint (0-based)
            
        Returns:
            float: Joint angle in degrees, or None if index is invalid
        """
        if 0 <= joint_index < self.num_dof:
            return math.degrees(self.angles[joint_index])
        return None


class KinematicWidget(QWidget):
    """
    Qt widget for visualizing the kinematic chain.
    
    This widget handles all the drawing operations for the kinematic chain,
    including joints (black circles), arms (gray lines), angle indicators
    (white lines), and the end effector (red circle).
    """
    
    def __init__(self, kinematic_chain):
        """
        Initialize the visualization widget.
        
        Args:
            kinematic_chain (KinematicChain): The kinematic chain to visualize
        """
        super().__init__()
        self.chain = kinematic_chain
        
        # Set fixed window size
        self.setFixedSize(800, 800)
        
        # Calculate center coordinates for coordinate transformation
        self.center_x = 400
        self.center_y = 400
        
        # Visual parameters
        self.joint_radius = 8  # Radius of joint circles
        self.arm_width = 4     # Width of arm lines
        self.angle_line_length = 12  # Length of angle indicator lines
        self.end_effector_radius = 6  # Radius of end effector circle
        
    def paintEvent(self, event):
        """
        Qt paint event handler - draws the entire kinematic chain.
        
        This method is called automatically by Qt whenever the widget needs
        to be redrawn. It renders all components of the kinematic chain.
        
        Args:
            event: Qt paint event (automatically provided)
        """
        # Initialize painter with anti-aliasing for smooth graphics
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Clear background to white
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # Draw all arm segments first (so they appear behind joints)
        self._draw_arms(painter)
        
        # Draw joint circles and angle indicators
        self._draw_joints(painter)
        
        # Draw end effector
        self._draw_end_effector(painter)
        
        # Draw information text
        self._draw_info_text(painter)
    
    def _draw_arms(self, painter):
        """
        Draw all arm segments as gray lines.
        
        Args:
            painter (QPainter): Qt painter object
        """
        # Set up pen for drawing arms
        pen = QPen(QColor(128, 128, 128), self.arm_width)
        painter.setPen(pen)
        
        # Draw line between each consecutive pair of joints
        for i in range(len(self.chain.joint_positions) - 1):
            # Get start position (current joint)
            start_x = self.center_x + self.chain.joint_positions[i][0]
            start_y = self.center_y + self.chain.joint_positions[i][1]
            
            # Get end position (next joint)
            end_x = self.center_x + self.chain.joint_positions[i + 1][0]
            end_y = self.center_y + self.chain.joint_positions[i + 1][1]
            
            # Draw the arm segment
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
    
    def _draw_joints(self, painter):
        """
        Draw joint circles and their angle indicators.
        
        Args:
            painter (QPainter): Qt painter object
        """
        # Set up brush and pen for joint circles (black filled circles)
        brush = QBrush(QColor(0, 0, 0))
        painter.setBrush(brush)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        
        # Draw each joint (excluding the end effector position)
        for i, (x, y) in enumerate(self.chain.joint_positions[:-1]):
            # Convert to screen coordinates
            joint_x = self.center_x + x
            joint_y = self.center_y + y
            
            # Draw joint as filled black circle
            painter.drawEllipse(
                int(joint_x - self.joint_radius), 
                int(joint_y - self.joint_radius),
                2 * self.joint_radius, 
                2 * self.joint_radius
            )
            
            # Draw angle indicator (white line showing current joint orientation)
            if i < len(self.chain.angles):
                self._draw_angle_indicator(painter, joint_x, joint_y, i)
    
    def _draw_angle_indicator(self, painter, joint_x, joint_y, joint_index):
        """
        Draw a white line indicating the current angle of a joint.
        
        Args:
            painter (QPainter): Qt painter object
            joint_x (float): X coordinate of joint center
            joint_y (float): Y coordinate of joint center
            joint_index (int): Index of the joint
        """
        # Calculate cumulative angle up to this joint
        cumulative_angle = sum(self.chain.angles[:joint_index + 1])
        
        # Calculate end point of angle indicator line
        line_end_x = joint_x + self.angle_line_length * math.cos(cumulative_angle)
        line_end_y = joint_y + self.angle_line_length * math.sin(cumulative_angle)
        
        # Set up pen for white angle indicator
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        
        # Draw the angle indicator line
        painter.drawLine(
            int(joint_x), int(joint_y), 
            int(line_end_x), int(line_end_y)
        )
    
    def _draw_end_effector(self, painter):
        """
        Draw the end effector as a red filled circle.
        
        Args:
            painter (QPainter): Qt painter object
        """
        # Set up brush and pen for end effector (red filled circle)
        brush = QBrush(QColor(255, 0, 0))
        painter.setBrush(brush)
        painter.setPen(QPen(QColor(255, 0, 0), 1))
        
        # Convert to screen coordinates
        ee_x = self.center_x + self.chain.end_effector_pos[0]
        ee_y = self.center_y + self.chain.end_effector_pos[1]
        
        # Draw end effector circle
        painter.drawEllipse(
            int(ee_x - self.end_effector_radius), 
            int(ee_y - self.end_effector_radius),
            2 * self.end_effector_radius, 
            2 * self.end_effector_radius
        )
    
    def _draw_info_text(self, painter):
        """
        Draw informational text showing current status.
        
        Args:
            painter (QPainter): Qt painter object
        """
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        
        # Calculate end effector position relative to center
        ee_x_rel = self.chain.end_effector_pos[0]
        ee_y_rel = self.chain.end_effector_pos[1]
        
        # Create info text
        info_text = f"DOF: {self.chain.num_dof} | End Effector: ({ee_x_rel:.1f}, {ee_y_rel:.1f})"
        
        # Draw text in top-left corner
        painter.drawText(10, 20, info_text)


class MainWindow(QMainWindow):
    """
    Main application window containing the kinematic chain simulation.
    
    This class manages the overall application, including the animation timer
    and demo sequence that rotates each joint through a full revolution.
    """
    
    def __init__(self, num_dof=3):
        """
        Initialize the main window and start the demo.
        
        Args:
            num_dof (int): Number of degrees of freedom for the kinematic chain
        """
        super().__init__()
        
        # Configure window
        self.setWindowTitle(f"Kinematic Actuator Chain - {num_dof} DOF")
        self.setFixedSize(820, 850)  # Slightly larger than widget for margins
        
        # Create kinematic chain model
        self.chain = KinematicChain(num_dof)
        
        # Set up UI layout
        self._setup_ui()
        
        # Animation parameters
        self.current_joint = 0  # Index of currently animating joint
        self.target_angle = 0   # Target angle for current animation
        self.animation_speed = 0.05  # Radians per frame
        self.is_animating = False    # Animation state flag
        
        # Set up animation timer (50 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_step)
        self.timer.start(20)  # 20ms = 50 FPS
        
        # Start the demo animation
        self.start_demo()
    
    def _setup_ui(self):
        """Set up the user interface layout."""
        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Create and add the kinematic visualization widget
        self.kinematic_widget = KinematicWidget(self.chain)
        layout.addWidget(self.kinematic_widget)
        
        # Set up the layout
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def start_demo(self):
        """
        Start the demonstration animation sequence.
        
        Initializes the demo to sequentially rotate each joint through
        a full 360-degree revolution, starting from the first joint.
        """
        print(f"Starting demo with {self.chain.num_dof} degrees of freedom")
        print("Initial angles:", [f"{math.degrees(a):.1f}°" for a in self.chain.angles])
        print("Demo sequence: Each joint will perform a full 360° rotation")
        print("-" * 60)
        
        # Initialize animation parameters
        self.current_joint = 0
        self.target_angle = self.chain.angles[0] + 2 * math.pi  # Add full rotation
        self.is_animating = True
        
        print(f"Starting rotation of Joint {self.current_joint + 1}")
    
    def animate_step(self):
        """
        Perform one step of the animation.
        
        This method is called by the QTimer at regular intervals to update
        the animation. It handles the smooth rotation of joints and progression
        through the demo sequence.
        """
        if not self.is_animating:
            return
        
        # Get current angle of the joint being animated
        current_angle = self.chain.angles[self.current_joint]
        
        # Check if we need to continue rotating toward target
        if abs(self.target_angle - current_angle) > self.animation_speed:
            # Calculate next angle step
            if self.target_angle > current_angle:
                new_angle = current_angle + self.animation_speed
            else:
                new_angle = current_angle - self.animation_speed
            
            # Update joint angle and refresh display
            self.chain.set_angle(self.current_joint, new_angle)
            self.kinematic_widget.update()
            
        else:
            # Target reached - complete current joint and move to next
            self.chain.set_angle(self.current_joint, self.target_angle)
            print(f"Joint {self.current_joint + 1}: Full rotation completed")
            
            # Move to next joint
            self.current_joint += 1
            
            if self.current_joint < self.chain.num_dof:
                # Set up animation for next joint
                self.target_angle = self.chain.angles[self.current_joint] + 2 * math.pi
                print(f"Starting rotation of Joint {self.current_joint + 1}")
            else:
                # All joints completed - end demo
                print("-" * 60)
                print("Demo completed! All joints have performed full rotations.")
                print("Final end effector position:", 
                      f"({self.chain.end_effector_pos[0]:.1f}, {self.chain.end_effector_pos[1]:.1f})")
                self.is_animating = False


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Kinematic Actuator Chain Simulation",
        epilog="Example: python kinematic_chain.py 5"
    )
    parser.add_argument(
        "num_dof", 
        nargs="?", 
        type=int, 
        default=3,
        help="Number of degrees of freedom (revolute joints). Default: 3"
    )
    return parser.parse_args()


def validate_arguments(args):
    """
    Validate command line arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Returns:
        bool: True if arguments are valid, False otherwise
    """
    if args.num_dof < 1:
        print("Error: Number of DOF must be at least 1.")
        return False
    
    if args.num_dof > 15:
        print("Warning: High number of DOF may make visualization cluttered.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            return False
    
    return True


def main():
    """
    Main application entry point.
    
    Handles argument parsing, validation, and application startup.
    """
    # Parse and validate command line arguments
    args = parse_arguments()
    
    if not validate_arguments(args):
        return 1
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow(args.num_dof)
    window.show()
    
    # Print startup information
    print(f"\n🤖 Kinematic Chain Simulation Started")
    print(f"   • Degrees of Freedom: {args.num_dof}")
    print(f"   • Window Size: 800x800 pixels")
    print(f"   • Animation Speed: 50 FPS")
    print(f"\n📖 Visual Legend:")
    print(f"   • Black circles: Revolute joints")
    print(f"   • White lines: Current joint angles")
    print(f"   • Gray lines: Arm segments")
    print(f"   • Red circle: End effector")
    print(f"\n💡 Close the window to exit the application.")
    
    # Start the Qt event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())