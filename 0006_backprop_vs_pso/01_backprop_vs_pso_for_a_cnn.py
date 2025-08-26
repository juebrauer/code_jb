#!/usr/bin/env python3
"""
CNN Training Comparison: Backpropagation vs Particle Swarm Optimization
This script trains a CNN on FashionMNIST using both traditional backpropagation
and Particle Swarm Optimization (PSO), then compares their performance.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time
import copy
from typing import Tuple, List
import random


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for FashionMNIST classification.
    Architecture:
    - Conv1: 1 -> 16 channels, 3x3 kernel
    - Conv2: 16 -> 32 channels, 3x3 kernel
    - FC1: Flattened features -> 128 neurons
    - FC2: 128 -> 10 classes (output)
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 16 output channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Second convolutional layer: 16 input channels, 32 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)
        # First fully connected layer: 32*7*7 = 1568 inputs, 128 outputs
        # (28x28 -> 14x14 after first pool -> 7x7 after second pool)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # Second fully connected layer: 128 inputs, 10 outputs (10 classes)
        self.fc2 = nn.Linear(128, 10)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """Forward pass through the network"""
        # First conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 32 * 7 * 7)
        # First FC layer with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc2(x)
        return x


class Particle:
    """
    Represents a single particle in the PSO algorithm.
    Each particle represents a complete set of neural network weights.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize a particle with random weights from the model.
        
        Args:
            model: The neural network model to get the weight structure from
        """
        # Flatten all model parameters into a single vector
        self.position = self._flatten_params(model)
        # Initialize velocity randomly (small values)
        self.velocity = np.random.randn(len(self.position)) * 0.01
        # Personal best position starts as current position
        self.best_position = self.position.copy()
        # Personal best fitness (lower is better for loss)
        self.best_fitness = float('inf')
        
    def _flatten_params(self, model: nn.Module) -> np.ndarray:
        """
        Flatten all model parameters into a single numpy array.
        
        Args:
            model: PyTorch model
            
        Returns:
            1D numpy array containing all model parameters
        """
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def update_velocity(self, global_best_position: np.ndarray, 
                       w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        Update particle velocity using PSO update rule.
        
        Args:
            global_best_position: Best position found by any particle in the swarm
            w: Inertia weight (controls exploration vs exploitation)
            c1: Cognitive parameter (attraction to personal best)
            c2: Social parameter (attraction to global best)
        """
        # Random factors for stochastic behavior
        r1 = np.random.random(len(self.position))
        r2 = np.random.random(len(self.position))
        
        # PSO velocity update equation:
        # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
        
        # Velocity clamping to prevent explosion
        max_velocity = 1.0
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
    
    def update_position(self):
        """Update particle position based on velocity."""
        self.position += self.velocity


class PSO:
    """
    Particle Swarm Optimization implementation for neural network training.
    """
    
    def __init__(self, model: nn.Module, n_particles: int = 30):
        """
        Initialize PSO optimizer.
        
        Args:
            model: The neural network model to optimize
            n_particles: Number of particles in the swarm
        """
        self.model = model
        self.n_particles = n_particles
        # Initialize swarm with random particles
        self.particles = [Particle(model) for _ in range(n_particles)]
        # Global best position (will be updated during optimization)
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_fitness = float('inf')
        
    def _set_model_params(self, position: np.ndarray):
        """
        Set model parameters from a flattened position vector.
        
        Args:
            position: 1D numpy array containing all parameters
        """
        idx = 0
        for param in self.model.parameters():
            param_shape = param.shape
            param_size = param.numel()
            # Reshape the relevant slice of the position vector
            param.data = torch.from_numpy(
                position[idx:idx+param_size].reshape(param_shape)
            ).float().to(param.device)
            idx += param_size
    
    def evaluate_fitness(self, position: np.ndarray, dataloader: DataLoader, 
                        device: torch.device) -> float:
        """
        Evaluate the fitness (loss) of a particle position.
        
        Args:
            position: Particle position (network weights)
            dataloader: Data to evaluate on
            device: Device to run evaluation on
            
        Returns:
            Average loss over the dataset
        """
        # Set model parameters to this position
        self._set_model_params(position)
        self.model.eval()
        
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def optimize_step(self, dataloader: DataLoader, device: torch.device):
        """
        Perform one PSO optimization step.
        
        Args:
            dataloader: Training data
            device: Device to run on
        """
        # Evaluate all particles
        for particle in self.particles:
            fitness = self.evaluate_fitness(particle.position, dataloader, device)
            
            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
        
        # Update velocities and positions
        for particle in self.particles:
            particle.update_velocity(self.global_best_position)
            particle.update_position()


def train_backprop(model: nn.Module, train_loader: DataLoader, 
                  test_loader: DataLoader, epochs: int, 
                  device: torch.device) -> Tuple[List[float], List[float], float]:
    """
    Train model using traditional backpropagation with Adam optimizer.
    
    Args:
        model: Neural network to train
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        Tuple of (train_losses, test_losses, training_time)
    """
    print("\n" + "="*60)
    print("TRAINING WITH BACKPROPAGATION")
    print("="*60)
    
    model = model.to(device)
    # Adam optimizer with default learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients from previous step
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate loss
            loss = criterion(output, target)
            # Backward pass (compute gradients)
            loss.backward()
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                # Get predictions
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate average test loss and accuracy
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, '
              f'Test Accuracy: {accuracy:.2f}%')

    training_time = time.time() - start_time
    return train_losses, test_losses, training_time


def train_pso(model: nn.Module, train_loader: DataLoader, 
             test_loader: DataLoader, epochs: int, 
             device: torch.device, n_particles: int = 20) -> Tuple[List[float], List[float], float]:
    """
    Train model using Particle Swarm Optimization.
    
    Args:
        model: Neural network to train
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        device: Device to train on
        n_particles: Number of particles in the swarm
        
    Returns:
        Tuple of (train_losses, test_losses, training_time)
    """
    print("\n" + "="*60)
    print("TRAINING WITH PARTICLE SWARM OPTIMIZATION")
    print("="*60)
    print(f"Number of particles: {n_particles}")
    
    model = model.to(device)
    pso = PSO(model, n_particles=n_particles)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Perform PSO optimization step
        pso.optimize_step(train_loader, device)
        
        # Set model to best found position
        pso._set_model_params(pso.global_best_position)
        
        # Evaluate on training set
        model.eval()
        train_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                train_loss += criterion(output, target).item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, '
              f'Test Accuracy: {accuracy:.2f}%')
    
    training_time = time.time() - start_time
    return train_losses, test_losses, training_time


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def main():
    """Main function to run the comparison experiment."""
    
    # Command line argument parser
    parser = argparse.ArgumentParser(
        description='Compare CNN training with Backprop vs PSO on FashionMNIST'
    )
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--train-samples', type=int, default=None,
                       help='Number of training samples to use (default: all 60000)')
    parser.add_argument('--test-samples', type=int, default=None,
                       help='Number of test samples to use (default: all 10000)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to train on (default: auto)')
    parser.add_argument('--pso-particles', type=int, default=20,
                       help='Number of particles for PSO (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\nUsing device: {device}")
    print(f"Random seed: {args.seed}")
    
    # Data preprocessing: Normalize to mean=0, std=1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load FashionMNIST dataset
    print("\nLoading FashionMNIST dataset...")
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Subset datasets if requested
    if args.train_samples is not None:
        train_indices = list(range(min(args.train_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        print(f"Using {len(train_dataset)} training samples")
    else:
        print(f"Using all {len(train_dataset)} training samples")
    
    if args.test_samples is not None:
        test_indices = list(range(min(args.test_samples, len(test_dataset))))
        test_dataset = Subset(test_dataset, test_indices)
        print(f"Using {len(test_dataset)} test samples")
    else:
        print(f"Using all {len(test_dataset)} test samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.epochs}")
    
    # Initialize two identical models with the same random weights
    print("\nInitializing models with identical random weights...")
    model_backprop = SimpleCNN()
    # Deep copy to ensure PSO starts with exact same weights
    model_pso = copy.deepcopy(model_backprop)
    
    # Verify that models start with identical weights
    for p1, p2 in zip(model_backprop.parameters(), model_pso.parameters()):
        assert torch.allclose(p1, p2), "Models don't start with identical weights!"
    print("Models initialized successfully with identical weights")
    
    # Train with backpropagation
    train_losses_bp, test_losses_bp, time_bp = train_backprop(
        model_backprop, train_loader, test_loader, args.epochs, device
    )
    
    # Train with PSO
    train_losses_pso, test_losses_pso, time_pso = train_pso(
        model_pso, train_loader, test_loader, args.epochs, device, args.pso_particles
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    # Evaluate backprop model
    loss_bp, acc_bp = evaluate_model(model_backprop, test_loader, device)
    print(f"\nBackpropagation Results:")
    print(f"  Final Test Loss:     {loss_bp:.4f}")
    print(f"  Final Test Accuracy: {acc_bp:.2f}%")
    print(f"  Training Time:       {time_bp:.2f} seconds")
    
    # Evaluate PSO model
    loss_pso, acc_pso = evaluate_model(model_pso, test_loader, device)
    print(f"\nPSO Results:")
    print(f"  Final Test Loss:     {loss_pso:.4f}")
    print(f"  Final Test Accuracy: {acc_pso:.2f}%")
    print(f"  Training Time:       {time_pso:.2f} seconds")
    
    # Comparison summary
    print("\n" + "-"*60)
    print("Summary:")
    print("-"*60)
    
    if acc_bp > acc_pso:
        print(f"Backpropagation achieved better accuracy ({acc_bp:.2f}% vs {acc_pso:.2f}%)")
        print(f"Improvement: {acc_bp - acc_pso:.2f} percentage points")
    elif acc_pso > acc_bp:
        print(f"PSO achieved better accuracy ({acc_pso:.2f}% vs {acc_bp:.2f}%)")
        print(f"Improvement: {acc_pso - acc_bp:.2f} percentage points")
    else:
        print(f"Both methods achieved similar accuracy ({acc_bp:.2f}%)")
    
    print(f"\nBackpropagation was {time_pso/time_bp:.2f}x faster than PSO")
    
    # Optional: Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot training losses
        ax1.plot(train_losses_bp, label='Backprop Train', color='blue', linestyle='-')
        ax1.plot(test_losses_bp, label='Backprop Test', color='blue', linestyle='--')
        ax1.plot(train_losses_pso, label='PSO Train', color='red', linestyle='-')
        ax1.plot(test_losses_pso, label='PSO Test', color='red', linestyle='--')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot final comparison
        methods = ['Backprop', 'PSO']
        accuracies = [acc_bp, acc_pso]
        losses = [loss_bp, loss_pso]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax2.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='green')
        ax2.bar(x + width/2, np.array(losses) * 100, width, label='Loss (×100)', color='orange')
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Value')
        ax2.set_title('Final Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=150)
        print("\nPlots saved to 'comparison_results.png'")
        plt.show()
        
    except ImportError:
        print("\nNote: Install matplotlib to visualize the results")


if __name__ == "__main__":
    main()