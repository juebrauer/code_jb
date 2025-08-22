#!/usr/bin/env python3
"""
Comparison: Backpropagation CNN (PyTorch/Adam) vs. Simple DeltaW Rule (gradient-free)

This script demonstrates and compares two different learning approaches for training
a Convolutional Neural Network (CNN):

1) Generates synthetic image data or uses MNIST dataset
2) Trains a CNN with classical backpropagation using Adam optimizer
3) Trains an identical CNN with an alternative gradient-free rule:
   - For each parameter, a DeltaW (change vector) is maintained
   - For each mini-batch, a step p := p + DeltaW is attempted
   - If loss decreases: keep the step and maintain the current DeltaW
   - If loss increases: discard the step and sample a new random DeltaW
4) Visualizes the learning progress (Train/Test accuracy and loss) and shows sample predictions

Example usage:
  python backprop_vs_deltaw_cnn.py --dataset mnist --epochs 10 --batch 64
  python backprop_vs_deltaw_cnn.py --dataset synthetic --size 28 --channels 1 --classes 5

Note: DeltaW is expected to be much slower for CNNs due to the large parameter space.
"""

import argparse
from dataclasses import dataclass
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --------------------------- CNN Architecture ---------------------------

class SimpleCNN(nn.Module):
    """
    A simple CNN architecture suitable for small images (28x28 or 32x32).
    
    Architecture:
    - Conv Block 1: Conv(in_channels->32) -> ReLU -> MaxPool
    - Conv Block 2: Conv(32->64) -> ReLU -> MaxPool
    - Classifier: Linear(flattened->128) -> ReLU -> Linear(128->num_classes)
    
    This is intentionally kept small to make DeltaW training somewhat feasible.
    """
    
    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        """
        Initialize the CNN architecture.
        
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
            input_size: Size of the square input image (28 for MNIST, 32 for CIFAR-like)
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        # First conv block: increases channels from input to 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by half
        
        # Second conv block: increases channels from 32 to 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by half again
        
        # Calculate the size after two pooling operations
        # After 2 max pools with stride 2: input_size -> input_size/2 -> input_size/4
        final_size = input_size // 4
        flattened_size = 64 * final_size * final_size
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization (only used during training)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Second convolutional block
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch
        
        # Classifier head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# --------------------------- Data Generation ---------------------------

def create_synthetic_dataset(n_samples=1000, img_size=28, n_channels=1, n_classes=5, noise_level=0.1):
    """
    Creates a synthetic image classification dataset for testing.
    
    Generates simple patterns (stripes, circles, squares) as different classes.
    This is useful for quick testing without downloading real datasets.
    
    Args:
        n_samples: Number of samples to generate
        img_size: Size of square images
        n_channels: Number of image channels
        n_classes: Number of classes
        noise_level: Amount of random noise to add
    
    Returns:
        Tuple of (images, labels) tensors
    """
    images = []
    labels = []
    
    for i in range(n_samples):
        # Randomly select a class
        class_idx = i % n_classes
        
        # Create base image
        img = torch.zeros(n_channels, img_size, img_size)
        
        # Generate different patterns for each class
        if class_idx == 0:  # Horizontal stripes
            for y in range(0, img_size, 4):
                img[:, y:min(y+2, img_size), :] = 1.0
                
        elif class_idx == 1:  # Vertical stripes
            for x in range(0, img_size, 4):
                img[:, :, x:min(x+2, img_size)] = 1.0
                
        elif class_idx == 2:  # Centered circle
            center = img_size // 2
            radius = img_size // 4
            y, x = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='ij')
            mask = ((x - center) ** 2 + (y - center) ** 2) < radius ** 2
            img[:, mask] = 1.0
            
        elif class_idx == 3:  # Centered square
            quarter = img_size // 4
            img[:, quarter:3*quarter, quarter:3*quarter] = 1.0
            
        else:  # Diagonal stripes
            for i in range(-img_size, img_size, 6):
                for j in range(img_size):
                    if 0 <= i+j < img_size and j < img_size:
                        img[:, j, i+j] = 1.0
        
        # Add noise
        if noise_level > 0:
            img += torch.randn_like(img) * noise_level
            img = torch.clamp(img, 0, 1)
        
        images.append(img)
        labels.append(class_idx)
    
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

def load_mnist_data(data_dir='./data'):
    """
    Loads the MNIST dataset using torchvision.
    
    MNIST consists of 28x28 grayscale images of handwritten digits (0-9).
    
    Args:
        data_dir: Directory to store/load the dataset
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    return train_dataset, test_dataset

# --------------------------- Training and Evaluation ---------------------------

def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Evaluates model performance on a dataset.
    
    Computes both loss and accuracy metrics.
    
    Args:
        model: The CNN model to evaluate
        loader: DataLoader containing the evaluation dataset
        device: Device to run evaluation on (CPU/CUDA)
    
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

@dataclass
class TrainResult:
    """
    Data class to store training results for comparison.
    
    Attributes:
        train_loss: List of training losses, one per epoch
        train_acc: List of training accuracies, one per epoch
        test_loss: List of test losses, one per epoch
        test_acc: List of test accuracies, one per epoch
        model: The trained model
    """
    train_loss: list[float]
    train_acc: list[float]
    test_loss: list[float]
    test_acc: list[float]
    model: nn.Module

# --------------------------- Backpropagation Training ---------------------------

def train_backprop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   device: torch.device, epochs: int = 10, lr: float = 1e-3) -> TrainResult:
    """
    Trains a CNN using standard backpropagation with Adam optimizer.
    
    Args:
        model: The CNN to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to train on (CPU/CUDA)
        epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
    
    Returns:
        TrainResult containing training history and final model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Print progress
        print(f"[Backprop] Epoch {epoch+1:02d}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")
    
    return TrainResult(train_losses, train_accs, test_losses, test_accs, model)

# --------------------------- DeltaW Training ---------------------------

class DeltaWCNNTrainer:
    """
    Implements gradient-free DeltaW optimization for CNNs.
    
    Note: This is expected to be very slow for CNNs due to the large number
    of parameters, but it demonstrates that the algorithm can work even
    for convolutional architectures.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, base_scale: float = 1e-3):
        """
        Initialize the DeltaW trainer for CNNs.
        
        Args:
            model: The CNN to train
            device: Device to train on
            base_scale: Base scaling factor for delta magnitudes
        """
        self.model = model.to(device)
        self.device = device
        self.base_scale = base_scale
        self.criterion = nn.CrossEntropyLoss()
        self.delta_ws: list[torch.Tensor] = []
        self._init_deltas()
    
    def _init_deltas(self):
        """
        Initialize delta vectors for all model parameters.
        
        For CNNs, we need to be careful with the scale of deltas for
        different layer types (conv vs. fully connected).
        """
        self.delta_ws = []
        
        for name, param in self.model.named_parameters():
            # Adaptive scaling based on parameter type and magnitude
            if 'conv' in name:
                # Convolutional layers: smaller deltas
                scale_factor = 0.5
            elif 'fc' in name or 'linear' in name:
                # Fully connected layers: normal deltas
                scale_factor = 1.0
            else:
                # Bias and other parameters
                scale_factor = 0.3
            
            # Estimate parameter scale
            p_std = param.data.std().item() if param.data.numel() > 1 else param.data.abs().mean().item()
            scale = self.base_scale * scale_factor * (p_std if p_std > 0 else 0.01)
            
            # Generate random delta
            delta = torch.randn_like(param.data) * scale
            self.delta_ws.append(delta)
    
    def _resample_deltas(self):
        """
        Resample all delta vectors when an update is rejected.
        """
        new_deltas = []
        
        for name, param in self.model.named_parameters():
            # Adaptive scaling based on parameter type
            if 'conv' in name:
                scale_factor = 0.5
            elif 'fc' in name or 'linear' in name:
                scale_factor = 1.0
            else:
                scale_factor = 0.3
            
            p_std = param.data.std().item() if param.data.numel() > 1 else param.data.abs().mean().item()
            scale = self.base_scale * scale_factor * (p_std if p_std > 0 else 0.01)
            
            delta = torch.randn_like(param.data) * scale
            new_deltas.append(delta)
        
        self.delta_ws = new_deltas
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 10) -> TrainResult:
        """
        Train the CNN using the DeltaW algorithm.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            epochs: Number of training epochs
        
        Returns:
            TrainResult containing training history and final model
        """
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []
        
        for epoch in range(epochs):
            self.model.train()
            accepted = 0
            total = 0
            
            # Process batches
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    # Compute loss before update
                    outputs_old = self.model(images)
                    loss_old = self.criterion(outputs_old, labels)
                    
                    # Save current parameters
                    old_params = [p.data.clone() for p in self.model.parameters()]
                    
                    # Apply delta updates
                    for param, delta in zip(self.model.parameters(), self.delta_ws):
                        param.data.add_(delta)
                    
                    # Compute loss after update
                    outputs_new = self.model(images)
                    loss_new = self.criterion(outputs_new, labels)
                    
                    total += 1
                    
                    # Accept or reject update
                    if loss_new.item() < loss_old.item():
                        # Accept: keep the update
                        accepted += 1
                    else:
                        # Reject: revert and resample
                        for param, old_data in zip(self.model.parameters(), old_params):
                            param.data.copy_(old_data)
                        self._resample_deltas()
            
            # Evaluate performance
            train_loss, train_acc = evaluate_model(self.model, train_loader, self.device)
            test_loss, test_acc = evaluate_model(self.model, test_loader, self.device)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            # Print progress
            acc_rate = accepted / max(1, total)
            print(f"[DeltaW] Epoch {epoch+1:02d}  AcceptRate: {acc_rate:.2f}  "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
                  f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")
        
        return TrainResult(train_losses, train_accs, test_losses, test_accs, self.model)

# --------------------------- Visualization ---------------------------

def make_comparison_plots(bp_result: TrainResult, dw_result: TrainResult, 
                         output_dir: Path, dataset_name: str):
    """
    Creates comprehensive comparison plots for both training methods.
    
    Generates:
    1. Loss curves (training and test)
    2. Accuracy curves (training and test)
    3. Combined comparison plot
    
    Args:
        bp_result: Results from backpropagation training
        dw_result: Results from DeltaW training
        output_dir: Directory to save plots
        dataset_name: Name of the dataset for plot titles
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = np.arange(1, len(bp_result.train_loss) + 1)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, bp_result.train_loss, 'b-', label='Backprop', linewidth=2)
    ax1.plot(epochs, dw_result.train_loss, 'r--', label='DeltaW', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss - {dataset_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, bp_result.test_loss, 'b-', label='Backprop', linewidth=2)
    ax2.plot(epochs, dw_result.test_loss, 'r--', label='DeltaW', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Test Loss - {dataset_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, bp_result.train_acc, 'b-', label='Backprop', linewidth=2)
    ax3.plot(epochs, dw_result.train_acc, 'r--', label='DeltaW', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title(f'Training Accuracy - {dataset_name}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Test Accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, bp_result.test_acc, 'b-', label='Backprop', linewidth=2)
    ax4.plot(epochs, dw_result.test_acc, 'r--', label='DeltaW', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title(f'Test Accuracy - {dataset_name}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cnn_comparison_plots.png', dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {output_dir.resolve()}")

# --------------------------- Main Program ---------------------------

def main():
    """
    Main function to run the CNN comparison experiment.
    """
    parser = argparse.ArgumentParser(
        description="Compare Backpropagation vs DeltaW for CNN training"
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'synthetic'],
                       help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset storage')
    
    # Synthetic data arguments
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of synthetic samples (if using synthetic)')
    parser.add_argument('--size', type=int, default=28,
                       help='Image size for synthetic data')
    parser.add_argument('--channels', type=int, default=1,
                       help='Number of channels for synthetic data')
    parser.add_argument('--classes', type=int, default=5,
                       help='Number of classes for synthetic data')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for backpropagation')
    parser.add_argument('--deltaw_scale', type=float, default=5e-4,
                       help='Base scale for DeltaW updates')
    
    # Other arguments
    parser.add_argument('--output', type=str, 
                       default='output_comparison_backprop_vs_deltaw_for_a_cnn',
                       help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load or create dataset
    print(f"\nPreparing dataset: {args.dataset}")
    
    if args.dataset == 'mnist':
        train_dataset, test_dataset = load_mnist_data(args.data_dir)
        n_channels = 1
        n_classes = 10
        input_size = 28
    else:  # synthetic
        print(f"Creating synthetic dataset with {args.n_samples} samples...")
        X_train, y_train = create_synthetic_dataset(
            n_samples=args.n_samples,
            img_size=args.size,
            n_channels=args.channels,
            n_classes=args.classes
        )
        X_test, y_test = create_synthetic_dataset(
            n_samples=args.n_samples // 5,  # 20% for testing
            img_size=args.size,
            n_channels=args.channels,
            n_classes=args.classes
        )
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        n_channels = args.channels
        n_classes = args.classes
        input_size = args.size
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    print(f"Dataset prepared: {len(train_dataset)} training samples, "
          f"{len(test_dataset)} test samples")
    
    # Create identical CNN models
    print(f"\nCreating CNN models (channels={n_channels}, classes={n_classes})...")
    
    base_model = SimpleCNN(in_channels=n_channels, num_classes=n_classes, input_size=input_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Total parameters in CNN: {total_params:,}")
    
    # Create two identical copies
    model_bp = SimpleCNN(in_channels=n_channels, num_classes=n_classes, input_size=input_size)
    model_bp.load_state_dict(base_model.state_dict())
    
    model_dw = SimpleCNN(in_channels=n_channels, num_classes=n_classes, input_size=input_size)
    model_dw.load_state_dict(base_model.state_dict())
    
    # Train with backpropagation
    print(f"\n{'='*60}")
    print("Training CNN with Backpropagation (Adam optimizer)")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}")
    
    bp_result = train_backprop(
        model_bp, train_loader, test_loader, 
        device, epochs=args.epochs, lr=args.lr
    )
    
    # Train with DeltaW
    print(f"\n{'='*60}")
    print("Training CNN with DeltaW (Gradient-free)")
    print(f"Delta scale: {args.deltaw_scale}")
    print(f"{'='*60}")
    
    dw_trainer = DeltaWCNNTrainer(model_dw, device, base_scale=args.deltaw_scale)
    dw_result = dw_trainer.train(train_loader, test_loader, epochs=args.epochs)
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON - CNN TRAINING RESULTS")
    print(f"{'='*60}")
    
    print(f"\nBackpropagation Performance:")
    print(f"  Final Training Loss: {bp_result.train_loss[-1]:.4f}")
    print(f"  Final Training Accuracy: {bp_result.train_acc[-1]:.2f}%")
    print(f"  Final Test Loss: {bp_result.test_loss[-1]:.4f}")
    print(f"  Final Test Accuracy: {bp_result.test_acc[-1]:.2f}%")
    
    print(f"\nDeltaW Performance:")
    print(f"  Final Training Loss: {dw_result.train_loss[-1]:.4f}")
    print(f"  Final Training Accuracy: {dw_result.train_acc[-1]:.2f}%")
    print(f"  Final Test Loss: {dw_result.test_loss[-1]:.4f}")
    print(f"  Final Test Accuracy: {dw_result.test_acc[-1]:.2f}%")
    
    print(f"\nRelative Performance:")
    
    # Accuracy comparison
    bp_test_acc = bp_result.test_acc[-1]
    dw_test_acc = dw_result.test_acc[-1]
    
    if bp_test_acc > dw_test_acc:
        diff = bp_test_acc - dw_test_acc
        print(f"  ✓ Backpropagation wins by {diff:.2f}% accuracy")
    else:
        diff = dw_test_acc - bp_test_acc
        print(f"  ✓ DeltaW wins by {diff:.2f}% accuracy")
    
    # Convergence speed analysis
    print(f"\nConvergence Analysis:")
    
    # Find epoch where 90% of final accuracy is reached
    bp_90_percent = 0.9 * bp_test_acc
    dw_90_percent = 0.9 * dw_test_acc
    
    bp_convergence_epoch = next((i+1 for i, acc in enumerate(bp_result.test_acc) 
                                 if acc >= bp_90_percent), args.epochs)
    dw_convergence_epoch = next((i+1 for i, acc in enumerate(dw_result.test_acc) 
                                 if acc >= dw_90_percent), args.epochs)
    
    print(f"  Epochs to reach 90% of final accuracy:")
    print(f"    Backprop: {bp_convergence_epoch} epochs")
    print(f"    DeltaW: {dw_convergence_epoch} epochs")
    
    if bp_convergence_epoch < dw_convergence_epoch:
        print(f"    → Backprop converges {dw_convergence_epoch - bp_convergence_epoch} epochs faster")
    elif dw_convergence_epoch < bp_convergence_epoch:
        print(f"    → DeltaW converges {bp_convergence_epoch - dw_convergence_epoch} epochs faster")
    else:
        print(f"    → Both methods converge at the same rate")
    
    # Parameter efficiency analysis
    print(f"\nParameter Efficiency:")
    print(f"  Total CNN parameters: {total_params:,}")
    print(f"  Parameters per class: {total_params // n_classes:,}")
    
    # Expected observations for CNNs with DeltaW
    print(f"\nExpected Behavior Analysis:")
    print(f"  • CNNs have many more parameters than MLPs ({total_params:,} total)")
    print(f"  • DeltaW explores parameter space randomly, which becomes")
    print(f"    exponentially harder as dimensionality increases")
    print(f"  • Backprop uses gradient information to navigate efficiently")
    print(f"  • For CNNs, DeltaW is expected to be significantly slower")
    
    print(f"{'='*60}")
    
    # Generate plots
    print(f"\nGenerating comparison plots...")
    output_dir = Path(args.output)
    make_comparison_plots(bp_result, dw_result, output_dir, args.dataset.upper())
    
    print("\nExperiment complete!")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()