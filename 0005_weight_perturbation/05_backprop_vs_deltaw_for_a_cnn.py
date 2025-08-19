#!/usr/bin/env python3
"""
Comparison: Backprop-CNN (PyTorch/Adam) vs. simple DeltaW-Rule (gradient-free)

1) Loads CIFAR-10 dataset (or alternatively Fashion-MNIST)
2) Trains a CNN with classical Backprop (Adam)
3) Trains an identical CNN with the DeltaW rule
4) Visualizes learning progress and shows confusion matrix

Usage examples:
  python cnn_backprop_vs_deltaw.py --dataset cifar10 --epochs 20 --batch 128
  python cnn_backprop_vs_deltaw.py --dataset fashion --epochs 15 --batch 256 --deltaw 5e-4

Note: CNN training with DeltaW is very inefficient and only for demonstration purposes!
"""

import argparse
from dataclasses import dataclass
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

# --------------------------- CNN Architecture ---------------------------

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10/Fashion-MNIST"""
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3x pooling: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TinyCNN(nn.Module):
    """Very small CNN for faster experiments"""
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --------------------------- Dataset Functions ---------------------------

def load_dataset(dataset_name='cifar10', data_path='./data', subset_size=None):
    """Loads CIFAR-10 or Fashion-MNIST"""
    
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                               download=True, transform=transform_test)
        in_channels = 3
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck')
        
    elif dataset_name == 'fashion':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True,
                                                     download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=data_path, train=False,
                                                    download=True, transform=transform)
        in_channels = 1
        classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Optional: Subset for faster tests
    if subset_size is not None:
        train_indices = torch.randperm(len(trainset))[:subset_size]
        test_indices = torch.randperm(len(testset))[:min(subset_size//5, len(testset))]
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)
    
    return trainset, testset, in_channels, classes


def evaluate_accuracy(model, loader, device):
    """Calculates accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


def get_predictions(model, loader, device):
    """Gets all predictions for confusion matrix"""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
    return np.array(all_targets), np.array(all_preds)


# --------------------------- Backprop Training ---------------------------

@dataclass
class TrainResult:
    train_acc: list[float]
    test_acc: list[float]
    train_loss: list[float]
    test_loss: list[float]
    model: nn.Module
    time_elapsed: float


def train_backprop(model, train_loader, test_loader, device, epochs=20, lr=1e-3):
    """Standard Backprop training with Adam"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_acc_hist, test_acc_hist = [], []
    train_loss_hist, test_loss_hist = [], []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Test evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = evaluate_accuracy(model, test_loader, device)
        
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
        
        print(f'[Backprop] Epoch {epoch+1:2d}/{epochs} - '
              f'Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}% - '
              f'Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')
    
    time_elapsed = time.time() - start_time
    return TrainResult(train_acc_hist, test_acc_hist, train_loss_hist, 
                      test_loss_hist, model, time_elapsed)


# --------------------------- DeltaW Training ---------------------------

class DeltaWTrainer:
    """DeltaW-rule trainer for CNNs"""
    
    def __init__(self, model, device, base_scale=1e-3, decay_factor=0.99):
        self.model = model.to(device)
        self.device = device
        self.base_scale = base_scale
        self.decay_factor = decay_factor
        self.criterion = nn.CrossEntropyLoss()
        self.delta_ws = []
        self.success_count = {}  # Counts successful updates per parameter
        self._init_deltas()
    
    def _init_deltas(self):
        """Initializes DeltaW for each parameter"""
        self.delta_ws = []
        for i, p in enumerate(self.model.parameters()):
            if p.requires_grad:
                # Intelligent scaling based on parameter type
                if len(p.shape) == 4:  # Conv layer
                    fan_in = p.shape[1] * p.shape[2] * p.shape[3]
                    scale = self.base_scale * (2.0 / fan_in) ** 0.5
                elif len(p.shape) == 2:  # FC layer
                    fan_in = p.shape[1]
                    scale = self.base_scale * (2.0 / fan_in) ** 0.5
                else:  # Bias or BatchNorm
                    scale = self.base_scale * 0.1
                
                delta = torch.randn_like(p.data) * scale
                self.delta_ws.append(delta.to(self.device))
                self.success_count[i] = 0
    
    def _resample_delta(self, idx, param):
        """Samples new delta for a specific parameter"""
        if len(param.shape) == 4:  # Conv layer
            fan_in = param.shape[1] * param.shape[2] * param.shape[3]
            scale = self.base_scale * (2.0 / fan_in) ** 0.5
        elif len(param.shape) == 2:  # FC layer
            fan_in = param.shape[1]
            scale = self.base_scale * (2.0 / fan_in) ** 0.5
        else:
            scale = self.base_scale * 0.1
        
        # Decay based on successes
        scale *= (self.decay_factor ** self.success_count[idx])
        
        self.delta_ws[idx] = torch.randn_like(param.data).to(self.device) * scale
    
    def train(self, train_loader, test_loader, epochs=20):
        """Training with DeltaW rule"""
        train_acc_hist, test_acc_hist = [], []
        train_loss_hist, test_loss_hist = [], []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            
            accepted_updates = 0
            total_updates = 0
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Calculate current loss
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss_before = self.criterion(outputs, labels)
                    
                    # Save old parameters
                    old_params = [p.data.clone() for p in self.model.parameters() if p.requires_grad]
                    
                    # Apply DeltaW
                    for i, (p, delta) in enumerate(zip(self.model.parameters(), self.delta_ws)):
                        if p.requires_grad:
                            p.data.add_(delta)
                    
                    # Calculate new loss
                    outputs_new = self.model(inputs)
                    loss_after = self.criterion(outputs_new, labels)
                    
                    total_updates += 1
                    
                    if loss_after.item() < loss_before.item():
                        # Update successful - keep it
                        accepted_updates += 1
                        running_loss += loss_after.item()
                        
                        # Update success counter
                        for i in range(len(self.delta_ws)):
                            self.success_count[i] += 1
                        
                        # For accuracy calculation
                        _, predicted = torch.max(outputs_new.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    else:
                        # Update rejected - restore old parameters
                        for i, p in enumerate(self.model.parameters()):
                            if p.requires_grad:
                                p.data.copy_(old_params[i])
                        
                        # Sample new deltas
                        for i, p in enumerate(self.model.parameters()):
                            if p.requires_grad:
                                self._resample_delta(i, p)
                                self.success_count[i] = max(0, self.success_count[i] - 1)
                        
                        running_loss += loss_before.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
            
            # Epoch statistics
            train_acc = 100 * correct / total
            train_loss = running_loss / len(train_loader)
            accept_rate = accepted_updates / max(1, total_updates)
            
            # Test evaluation
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    test_loss += self.criterion(outputs, labels).item()
            
            test_loss = test_loss / len(test_loader)
            test_acc = evaluate_accuracy(self.model, test_loader, self.device)
            
            train_acc_hist.append(train_acc)
            test_acc_hist.append(test_acc)
            train_loss_hist.append(train_loss)
            test_loss_hist.append(test_loss)
            
            print(f'[DeltaW] Epoch {epoch+1:2d}/{epochs} - '
                  f'Accept Rate: {accept_rate:.2%} - '
                  f'Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}% - '
                  f'Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')
        
        time_elapsed = time.time() - start_time
        return TrainResult(train_acc_hist, test_acc_hist, train_loss_hist, 
                          test_loss_hist, self.model, time_elapsed)


# --------------------------- Visualization ---------------------------

def plot_results(bp_result, dw_result, classes, test_loader, device, output_dir):
    """Creates comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = np.arange(1, len(bp_result.train_acc) + 1)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training Accuracy
    axes[0, 0].plot(epochs, bp_result.train_acc, 'b-', label='Backprop', linewidth=2)
    axes[0, 0].plot(epochs, dw_result.train_acc, 'r-', label='DeltaW', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Training Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test Accuracy
    axes[0, 1].plot(epochs, bp_result.test_acc, 'b-', label='Backprop', linewidth=2)
    axes[0, 1].plot(epochs, dw_result.test_acc, 'r-', label='DeltaW', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Loss
    axes[1, 0].plot(epochs, bp_result.train_loss, 'b-', label='Backprop', linewidth=2)
    axes[1, 0].plot(epochs, dw_result.train_loss, 'r-', label='DeltaW', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test Loss
    axes[1, 1].plot(epochs, bp_result.test_loss, 'b-', label='Backprop', linewidth=2)
    axes[1, 1].plot(epochs, dw_result.test_loss, 'r-', label='DeltaW', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Test Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('CNN Training: Backprop vs. DeltaW Rule', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Confusion matrices (only if test set is large enough)
    if len(test_loader.dataset) >= 10:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Backprop confusion matrix
        y_true_bp, y_pred_bp = get_predictions(bp_result.model, test_loader, device)
        cm_bp = confusion_matrix(y_true_bp, y_pred_bp)
        
        sns.heatmap(cm_bp, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=classes, yticklabels=classes)
        axes[0].set_title(f'Backprop (Acc: {bp_result.test_acc[-1]:.1f}%)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # DeltaW confusion matrix
        y_true_dw, y_pred_dw = get_predictions(dw_result.model, test_loader, device)
        cm_dw = confusion_matrix(y_true_dw, y_pred_dw)
        
        sns.heatmap(cm_dw, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                    xticklabels=classes, yticklabels=classes)
        axes[1].set_title(f'DeltaW (Acc: {dw_result.test_acc[-1]:.1f}%)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrices.png', dpi=100, bbox_inches='tight')
        plt.show()
    else:
        print(f"\nSkipping confusion matrices (test set too small: {len(test_loader.dataset)} samples)")
    
    print(f"\nPlots saved in: {output_dir.resolve()}")
    print(f"\nTime comparison:")
    print(f"  Backprop: {bp_result.time_elapsed:.2f} seconds")
    print(f"  DeltaW:   {dw_result.time_elapsed:.2f} seconds")
    print(f"  Factor:   {dw_result.time_elapsed/bp_result.time_elapsed:.2f}x slower")


# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description='CNN Training: Backprop vs DeltaW')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'fashion'],
                       help='Dataset (cifar10 or fashion)')
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'tiny'],
                       help='CNN architecture (simple or tiny)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for backprop')
    parser.add_argument('--deltaw', type=float, default=1e-3, 
                       help='Base scaling for DeltaW steps')
    parser.add_argument('--subset', type=int, default=None,
                       help='Subset size for faster tests (e.g. 5000)')
    parser.add_argument('--output', type=str, default='cnn_outputs',
                       help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device (auto, cpu or cuda)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Device with fallback
    if args.device == 'auto':
        if torch.cuda.is_available():
            # Test if CUDA actually works
            try:
                test_tensor = torch.randn(1, 3, 32, 32).cuda()
                _ = test_tensor * 2  # Simple operation to test CUDA
                device = torch.device('cuda')
                print("CUDA device detected and functional")
            except (RuntimeError, torch.cuda.CudaError) as e:
                print(f"CUDA available but not functional: {e}")
                print("Falling back to CPU...")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset.upper()}")
    if args.subset:
        print(f"Using subset with {args.subset} training samples")
    
    # Load data
    trainset, testset, in_channels, classes = load_dataset(
        args.dataset, subset_size=args.subset
    )
    
    train_loader = DataLoader(trainset, batch_size=args.batch, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.batch, 
                            shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    # Warn if dataset is too small
    if len(trainset) < 100:
        print("\n⚠️  WARNING: Very small training set! Results will not be meaningful.")
        print("   Recommended: Use at least --subset 1000 for basic testing")
        print("                or --subset 5000 for meaningful comparisons\n")
    
    # Create models with identical initialization
    if args.model == 'simple':
        base_model = SimpleCNN(num_classes=10, in_channels=in_channels)
    else:
        base_model = TinyCNN(num_classes=10, in_channels=in_channels)
    
    # Clone for both training methods
    model_bp = SimpleCNN(num_classes=10, in_channels=in_channels) if args.model == 'simple' \
               else TinyCNN(num_classes=10, in_channels=in_channels)
    model_bp.load_state_dict(base_model.state_dict())
    
    model_dw = SimpleCNN(num_classes=10, in_channels=in_channels) if args.model == 'simple' \
               else TinyCNN(num_classes=10, in_channels=in_channels)
    model_dw.load_state_dict(base_model.state_dict())
    
    # Number of parameters
    num_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    print("\n" + "="*60)
    print("Starting Backprop training...")
    print("="*60)
    bp_result = train_backprop(model_bp, train_loader, test_loader, 
                               device, epochs=args.epochs, lr=args.lr)
    
    print("\n" + "="*60)
    print("Starting DeltaW training...")
    print("="*60)
    dw_trainer = DeltaWTrainer(model_dw, device, base_scale=args.deltaw)
    dw_result = dw_trainer.train(train_loader, test_loader, epochs=args.epochs)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Backprop - Final test accuracy: {bp_result.test_acc[-1]:.2f}%")
    print(f"DeltaW   - Final test accuracy: {dw_result.test_acc[-1]:.2f}%")
    print(f"Difference: {bp_result.test_acc[-1] - dw_result.test_acc[-1]:.2f}%")
    
    # Visualization
    plot_results(bp_result, dw_result, classes, test_loader, device, args.output)


if __name__ == "__main__":
    main()