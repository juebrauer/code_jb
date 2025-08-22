#!/usr/bin/env python3
"""
Comparison: Backpropagation MLP (PyTorch/Adam) vs. Simple DeltaW Rule (gradient-free)

This script demonstrates and compares two different learning approaches for training
a Multi-Layer Perceptron (MLP) neural network:

1) Generates training/test data for a nonlinear target function f(x,y,z)
2) Trains an MLP with classical backpropagation using Adam optimizer
3) Trains an identical MLP with an alternative gradient-free rule:
   - For each parameter, a DeltaW (change vector) is maintained
   - For each mini-batch, a step p := p + DeltaW is attempted
   - If loss decreases: keep the step and maintain the current DeltaW
   - If loss increases: discard the step and sample a new random DeltaW
4) Visualizes the learning progress (Train/Test MSE) and shows a scatter plot
   (predictions vs. ground truth) at the end

Example usage:
  python backprop_vs_deltaw_mlp.py --train 5000 --test 1000 --epochs 25 --batch 256 --hidden 64

Note: For larger runs (e.g., 10k+ samples / 50 epochs), execution time may be significant depending on CPU.
"""

import argparse
from dataclasses import dataclass
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --------------------------- Helper Functions ---------------------------

def target_function(X: torch.Tensor) -> torch.Tensor:
    """
    Defines a complex nonlinear function f(x,y,z) with cosine and power terms.
    This function serves as the ground truth that the neural networks will learn to approximate.
    
    The function combines multiple nonlinear components:
    - Cosine terms for periodic behavior
    - Polynomial terms for smooth nonlinearity
    - Interaction terms (x*y*z) for feature coupling
    
    Args:
        X: Input tensor of shape (N, 3) where N is batch size
           Each row contains (x, y, z) coordinates
    
    Returns:
        Output tensor of shape (N, 1) containing function values
    """
    # Extract individual coordinate columns from input tensor
    x = X[:, 0]  # First column: x coordinates
    y = X[:, 1]  # Second column: y coordinates
    z = X[:, 2]  # Third column: z coordinates
    
    # Compute complex nonlinear function with multiple components
    f = (
        torch.cos(1.5 * x)          # Periodic component in x
        + 0.5 * (y ** 2)            # Quadratic term in y
        - 0.3 * (z ** 3)            # Cubic term in z (negative contribution)
        + 0.1 * x * y * z           # Three-way interaction term
        + 0.2 * torch.cos(y * z)    # Periodic interaction between y and z
        + 0.05 * (x ** 2) * z       # Quadratic-linear interaction
    )
    
    # Add dimension to match expected output shape (N, 1)
    return f.unsqueeze(1)


def make_dataset(n_samples: int, noise_std: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a synthetic dataset by sampling random inputs and computing the target function.
    
    This function creates training/test data by:
    1. Sampling random input points uniformly from a 3D cube
    2. Computing the true function values
    3. Optionally adding Gaussian noise to simulate real-world measurement errors
    
    Args:
        n_samples: Number of data points to generate
        noise_std: Standard deviation of Gaussian noise to add to outputs
                  (default: 0.05 for small noise, set to 0 for noiseless data)
    
    Returns:
        Tuple of (X, y) where:
        - X: Input features tensor of shape (n_samples, 3)
        - y: Target values tensor of shape (n_samples, 1)
    """
    # Generate random input points uniformly distributed in [-2, 2]³
    X = torch.empty(n_samples, 3).uniform_(-2.0, 2.0)
    
    # Compute true function values for all input points
    y = target_function(X)
    
    # Add Gaussian noise to outputs if specified (simulates measurement error)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    
    return X, y


def create_mlp(input_dim=3, hidden=64, depth=2) -> nn.Module:
    """
    Creates a Multi-Layer Perceptron (MLP) with configurable architecture.
    
    The network architecture follows this pattern:
    - Input layer: input_dim neurons
    - Hidden layers: 'depth' layers, each with 'hidden' neurons
    - Activation: Tanh after each hidden layer (smooth, bounded nonlinearity)
    - Output layer: Single neuron for regression
    
    Args:
        input_dim: Number of input features (default: 3 for x,y,z)
        hidden: Number of neurons in each hidden layer (default: 64)
        depth: Number of hidden layers (default: 2)
    
    Returns:
        nn.Sequential model containing the complete MLP architecture
    """
    # Start with input layer transforming input_dim -> hidden dimensions
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.Tanh()]
    
    # Add additional hidden layers (each maintains 'hidden' dimensions)
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    
    # Final output layer: hidden -> 1 (single regression output)
    # No activation here as we want unbounded output for regression
    layers += [nn.Linear(hidden, 1)]
    
    # Combine all layers into a sequential model
    return nn.Sequential(*layers)


def evaluate_mse(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluates the Mean Squared Error (MSE) of a model on a dataset.
    
    This function:
    1. Sets the model to evaluation mode (disables dropout, batch norm updates)
    2. Iterates through all batches without gradient computation
    3. Accumulates the squared errors
    4. Returns the average MSE across all samples
    
    Args:
        model: The neural network model to evaluate
        loader: DataLoader containing the dataset to evaluate on
    
    Returns:
        Float value representing the average MSE across all samples
    """
    # Set model to evaluation mode (important for dropout, batch norm, etc.)
    model.eval()
    
    # Initialize accumulators for loss and sample count
    loss_sum = 0.0
    n = 0
    
    # Disable gradient computation for efficiency during evaluation
    with torch.no_grad():
        for xb, yb in loader:
            # Forward pass: compute predictions
            pred = model(xb)
            
            # Accumulate sum of squared errors (not mean yet)
            loss_sum += nn.functional.mse_loss(pred, yb, reduction="sum").item()
            
            # Count total number of samples
            n += yb.numel()
    
    # Return mean squared error
    return loss_sum / n

# --------------------------- Backpropagation Training ---------------------------

@dataclass
class TrainResult:
    """
    Data class to store training results for easy comparison and visualization.
    
    Attributes:
        train_mse: List of training MSE values, one per epoch
        test_mse: List of test MSE values, one per epoch
        model: The trained model after completion
    """
    train_mse: list[float]
    test_mse: list[float]
    model: nn.Module


def train_backprop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   epochs: int = 25, lr: float = 1e-3) -> TrainResult:
    """
    Trains a neural network using standard backpropagation with Adam optimizer.
    
    This implements the classical deep learning training loop:
    1. Forward pass: compute predictions
    2. Compute loss (MSE for regression)
    3. Backward pass: compute gradients via automatic differentiation
    4. Update weights using Adam optimizer
    
    Args:
        model: The neural network to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs (default: 25)
        lr: Learning rate for Adam optimizer (default: 0.001)
    
    Returns:
        TrainResult containing training history and final model
    """
    # Define loss function (Mean Squared Error for regression)
    criterion = nn.MSELoss()
    
    # Initialize Adam optimizer (adaptive learning rates, momentum)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store training history for visualization
    train_hist, test_hist = [], []
    
    # Main training loop over epochs
    for ep in range(epochs):
        # Set model to training mode (enables dropout, batch norm updates)
        model.train()
        
        # Accumulators for computing epoch-level statistics
        run = 0.0  # Running sum of losses
        cnt = 0    # Running count of samples
        
        # Iterate over mini-batches
        for xb, yb in train_loader:
            # Zero out gradients from previous iteration
            # set_to_none=True is more memory efficient than zero_()
            opt.zero_grad(set_to_none=True)
            
            # Forward pass: compute predictions
            pred = model(xb)
            
            # Compute loss for this batch
            loss = criterion(pred, yb)
            
            # Backward pass: compute gradients via automatic differentiation
            loss.backward()
            
            # Update model parameters using computed gradients
            opt.step()
            
            # Accumulate statistics for epoch-level reporting
            run += loss.item() * yb.size(0)  # Total loss (weighted by batch size)
            cnt += yb.size(0)                # Total samples seen
        
        # Record epoch-level metrics
        train_mse = run / cnt  # Average training loss for this epoch
        test_mse = evaluate_mse(model, test_loader)  # Test set performance
        
        train_hist.append(train_mse)
        test_hist.append(test_mse)
        
        # Print progress information for each epoch (similar to DeltaW output)
        print(f"[Backprop] Ep {ep+1:02d}  TrainMSE={train_mse:.4f}  TestMSE={test_mse:.4f}")
    
    return TrainResult(train_hist, test_hist, model)

# --------------------------- Alternative DeltaW Rule ---------------------------

class DeltaWTrainer:
    """
    Implements a gradient-free optimization algorithm based on random weight perturbations.
    
    This algorithm is inspired by evolutionary strategies and random search methods:
    - Each parameter maintains a "delta" vector (direction of change)
    - At each step, parameters are updated by adding their delta
    - If the loss improves, keep the update and the same delta
    - If the loss worsens, revert the update and sample a new random delta
    
    This approach doesn't require gradient computation, making it:
    - Simpler to implement (no backpropagation needed)
    - Potentially more robust to non-smooth loss landscapes
    - Generally less efficient than gradient-based methods
    """
    
    def __init__(self, model: nn.Module, base_scale: float = 1e-3):
        """
        Initialize the DeltaW trainer.
        
        Args:
            model: The neural network to train
            base_scale: Base scaling factor for delta magnitudes
                       (controls the step size relative to parameter scale)
        """
        self.model = model
        self.base_scale = base_scale
        self.criterion = nn.MSELoss()
        self.delta_ws: list[torch.Tensor] = []
        
        # Initialize delta vectors for all parameters
        self._init_deltas()

    def _init_deltas(self):
        """
        Initializes delta (change) vectors for all model parameters.
        
        The magnitude of deltas is scaled based on:
        - The base_scale hyperparameter
        - The current scale of each parameter (adaptive scaling)
        
        This adaptive scaling helps maintain appropriate step sizes for
        parameters of different magnitudes (e.g., weights vs. biases).
        """
        self.delta_ws = []
        
        for p in self.model.parameters():
            # Estimate the scale of this parameter
            # Use std for matrices, mean absolute value for vectors/scalars
            if p.data.numel() > 1:
                p_std = float(p.data.std().item())
            else:
                p_std = float(p.data.abs().mean().item())
            
            # Scale the delta based on parameter magnitude
            # This ensures step sizes are proportional to parameter scale
            scale = self.base_scale * (p_std if p_std > 0 else 1.0)
            
            # Sample random delta from normal distribution
            self.delta_ws.append(torch.randn_like(p.data) * scale)

    def _resample_deltas(self):
        """
        Resamples all delta vectors with new random values.
        
        Called when a proposed update increases the loss, this method
        generates new random search directions while maintaining
        appropriate scaling for each parameter.
        """
        new = []
        
        for p in self.model.parameters():
            # Recompute scale based on current parameter values
            # (parameters may have changed scale during training)
            if p.data.numel() > 1:
                p_std = float(p.data.std().item())
            else:
                p_std = float(p.data.abs().mean().item())
            
            scale = self.base_scale * (p_std if p_std > 0 else 1.0)
            
            # Generate new random delta
            new.append(torch.randn_like(p.data) * scale)
        
        self.delta_ws = new

    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 25):
        """
        Trains the model using the DeltaW gradient-free optimization algorithm.
        
        The training process for each batch:
        1. Compute current loss
        2. Apply delta updates to all parameters
        3. Compute new loss
        4. If improved: keep changes and deltas
        5. If not improved: revert changes and sample new deltas
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            epochs: Number of training epochs
        
        Returns:
            TrainResult containing training history and final model
        """
        train_hist, test_hist = [], []
        
        for ep in range(epochs):
            # Set model to training mode
            self.model.train()
            
            # Track acceptance rate (useful for monitoring algorithm behavior)
            accepted = 0  # Number of accepted updates
            total = 0     # Total number of attempts
            
            # Process each mini-batch
            for xb, yb in train_loader:
                with torch.no_grad():
                    # Compute loss before update
                    loss_old = self.criterion(self.model(xb), yb)
                    
                    # Store current parameters (in case we need to revert)
                    old_params = [p.data.clone() for p in self.model.parameters()]
                    
                    # Apply candidate update: add delta to each parameter
                    for p, d in zip(self.model.parameters(), self.delta_ws):
                        p.add_(d)  # In-place addition for efficiency
                    
                    # Compute loss after update
                    loss_new = self.criterion(self.model(xb), yb)
                    
                    total += 1
                    
                    # Decision: accept or reject the update
                    if loss_new.item() < loss_old.item():
                        # Success! Loss decreased, keep the update
                        accepted += 1
                        # Keep current deltas for next iteration (momentum-like effect)
                    else:
                        # Failure: loss increased, revert the update
                        for p, old in zip(self.model.parameters(), old_params):
                            p.data.copy_(old)
                        
                        # Sample new random search directions
                        self._resample_deltas()
            
            # Evaluate performance on full datasets
            train_hist.append(evaluate_mse(self.model, train_loader))
            test_hist.append(evaluate_mse(self.model, test_loader))
            
            # Print progress information
            acc_rate = accepted / max(1, total)
            print(f"[DeltaW] Ep {ep+1:02d}  AccRate={acc_rate:.2f}  "
                  f"TrainMSE={train_hist[-1]:.4f}  TestMSE={test_hist[-1]:.4f}")
        
        return TrainResult(train_hist, test_hist, self.model)

# --------------------------- Plotting and Visualization ---------------------------

def make_plots(bp: TrainResult, alt: TrainResult, outdir: Path):
    """
    Creates comparison plots between backpropagation and DeltaW training results.
    
    Generates three plots:
    1. Training loss over epochs (comparing both methods)
    2. Test loss over epochs (comparing both methods)
    3. Scatter plot of predictions vs. true values on test set
    
    Args:
        bp: Training results from backpropagation
        alt: Training results from DeltaW algorithm
        outdir: Directory path to save the generated plots
    """
    # Create output directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create x-axis for epoch plots
    epochs = np.arange(1, len(bp.train_mse) + 1)

    # Plot 1: Training Loss Comparison
    plt.figure()
    plt.plot(epochs, bp.train_mse, label="Backprop (Train)")
    plt.plot(epochs, alt.train_mse, label="DeltaW Rule (Train)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Error Over Epochs")
    plt.legend()
    plt.savefig(outdir / "train_loss_comparison.png", bbox_inches="tight")

    # Plot 2: Test Loss Comparison
    plt.figure()
    plt.plot(epochs, bp.test_mse, label="Backprop (Test)")
    plt.plot(epochs, alt.test_mse, label="DeltaW Rule (Test)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Test Error Over Epochs")
    plt.legend()
    plt.savefig(outdir / "test_loss_comparison.png", bbox_inches="tight")

    # Plot 3: Predictions vs True Values (Scatter Plot)
    def preds_on(model: nn.Module, X: torch.Tensor) -> np.ndarray:
        """
        Helper function to get model predictions on a dataset.
        
        Args:
            model: Trained model
            X: Input features
        
        Returns:
            Numpy array of predictions
        """
        model.eval()
        with torch.no_grad():
            return model(X).cpu().numpy().ravel()

    plt.figure()
    
    # Generate fresh test data for fair comparison
    # Both models evaluated on the same test points
    X_test, y_test = make_dataset(2000, noise_std=0.05)
    y_true = y_test.numpy().ravel()
    
    # Get predictions from both models
    pred_bp = preds_on(bp.model, X_test)
    pred_alt = preds_on(alt.model, X_test)
    
    # Create scatter plot with transparency for overlapping points
    plt.scatter(y_true, pred_bp, s=6, alpha=0.6, label="Backprop")
    plt.scatter(y_true, pred_alt, s=6, alpha=0.6, label="DeltaW Rule")
    plt.xlabel("True (y)")
    plt.ylabel("Predicted (ŷ)")
    plt.title("Predictions vs. True Values (Test Data)")
    plt.legend()
    plt.savefig(outdir / "pred_vs_true_test.png", bbox_inches="tight")

    print(f"Plots saved in: {outdir.resolve()}")

# --------------------------- Main Entry Point ---------------------------

def main():
    """
    Main function that orchestrates the entire experiment:
    1. Parses command-line arguments
    2. Generates datasets
    3. Creates and initializes models
    4. Trains models using both methods
    5. Generates comparison plots
    """
    # Set up command-line argument parser
    p = argparse.ArgumentParser(
        description="Compare backpropagation vs. gradient-free DeltaW learning for MLPs"
    )
    
    # Dataset arguments
    p.add_argument("--train", type=int, default=5000, 
                   help="Number of training samples to generate")
    p.add_argument("--test", type=int, default=1000, 
                   help="Number of test samples to generate")
    
    # Training arguments
    p.add_argument("--epochs", type=int, default=25, 
                   help="Number of training epochs")
    p.add_argument("--batch", type=int, default=256, 
                   help="Batch size for mini-batch training")
    
    # Model architecture arguments
    p.add_argument("--hidden", type=int, default=64, 
                   help="Number of hidden units per layer")
    p.add_argument("--depth", type=int, default=2, 
                   help="Number of hidden layers")
    
    # Optimization arguments
    p.add_argument("--lr", type=float, default=1e-3, 
                   help="Learning rate for backpropagation (Adam)")
    p.add_argument("--deltaw", type=float, default=1e-3, 
                   help="Base scale for DeltaW step sizes")
    
    # Output arguments
    p.add_argument("--out", type=str, default="output_comparison_backprop_vs_deltaw_for_a_mlp", 
                   help="Output directory for plots")
    
    # Reproducibility
    p.add_argument("--seed", type=int, default=0, 
                   help="Random seed for reproducibility")
    
    args = p.parse_args()

    # Set random seeds for reproducibility across all libraries
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Generate synthetic datasets
    print(f"Generating {args.train} training samples and {args.test} test samples...")
    X_train, y_train = make_dataset(args.train, noise_std=0.05)
    X_test, y_test = make_dataset(args.test, noise_std=0.05)

    # Create DataLoaders for efficient batch processing
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=args.batch, 
        shuffle=True  # Shuffle training data for better convergence
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), 
        batch_size=args.batch, 
        shuffle=False  # Don't shuffle test data (not necessary)
    )

    # Create base model with specified architecture
    print(f"Creating MLPs with {args.depth} hidden layers of {args.hidden} units each...")
    base = create_mlp(hidden=args.hidden, depth=args.depth)
    
    # Create two identical copies for fair comparison
    # Both start from the same initial weights
    model_bp = create_mlp(hidden=args.hidden, depth=args.depth)
    model_bp.load_state_dict(base.state_dict())  # Copy weights from base
    
    model_alt = create_mlp(hidden=args.hidden, depth=args.depth)
    model_alt.load_state_dict(base.state_dict())  # Copy weights from base

    # Train using backpropagation
    print(f"\nTraining with Backpropagation (Adam, lr={args.lr})...")
    bp_res = train_backprop(model_bp, train_loader, test_loader, 
                            epochs=args.epochs, lr=args.lr)
    print(f"Backprop final: Train MSE = {bp_res.train_mse[-1]:.4f}, "
          f"Test MSE = {bp_res.test_mse[-1]:.4f}")

    # Train using alternative DeltaW rule
    print(f"\nTraining with DeltaW Rule (scale={args.deltaw})...")
    alt_trainer = DeltaWTrainer(model_alt, base_scale=args.deltaw)
    alt_res = alt_trainer.train(train_loader, test_loader, epochs=args.epochs)
    print(f"DeltaW final: Train MSE = {alt_res.train_mse[-1]:.4f}, "
          f"Test MSE = {alt_res.test_mse[-1]:.4f}")

    # Final comparison and analysis
    print("\n" + "="*60)
    print("FINAL COMPARISON - BACKPROP vs DELTAW")
    print("="*60)
    
    # Calculate performance metrics
    bp_train_final = bp_res.train_mse[-1]
    bp_test_final = bp_res.test_mse[-1]
    dw_train_final = alt_res.train_mse[-1]
    dw_test_final = alt_res.test_mse[-1]
    
    # Performance comparison
    print(f"\nBackpropagation Performance:")
    print(f"  Final Training MSE: {bp_train_final:.4f}")
    print(f"  Final Test MSE:     {bp_test_final:.4f}")
    
    print(f"\nDeltaW Rule Performance:")
    print(f"  Final Training MSE: {dw_train_final:.4f}")
    print(f"  Final Test MSE:     {dw_test_final:.4f}")
    
    # Relative comparison
    print(f"\nRelative Performance (lower is better):")
    
    # Training performance comparison
    if bp_train_final < dw_train_final:
        improvement = ((dw_train_final - bp_train_final) / dw_train_final) * 100
        print(f"  Training: Backprop is {improvement:.1f}% better")
    else:
        improvement = ((bp_train_final - dw_train_final) / bp_train_final) * 100
        print(f"  Training: DeltaW is {improvement:.1f}% better")
    
    # Test performance comparison
    if bp_test_final < dw_test_final:
        improvement = ((dw_test_final - bp_test_final) / dw_test_final) * 100
        print(f"  Test:     Backprop is {improvement:.1f}% better")
    else:
        improvement = ((bp_test_final - dw_test_final) / bp_test_final) * 100
        print(f"  Test:     DeltaW is {improvement:.1f}% better")
    
    # Overall winner determination
    print(f"\nOverall Winner:")
    if bp_test_final < dw_test_final:
        print(f"  ✓ Backpropagation wins with {bp_test_final:.4f} test MSE")
        print(f"    (DeltaW achieved {dw_test_final:.4f} test MSE)")
    else:
        print(f"  ✓ DeltaW Rule wins with {dw_test_final:.4f} test MSE")
        print(f"    (Backprop achieved {bp_test_final:.4f} test MSE)")
    
    # Check for overfitting
    print(f"\nOverfitting Analysis:")
    bp_overfit = bp_test_final - bp_train_final
    dw_overfit = dw_test_final - dw_train_final
    print(f"  Backprop gap (test-train): {bp_overfit:.4f}")
    print(f"  DeltaW gap (test-train):   {dw_overfit:.4f}")
    
    if abs(bp_overfit) < abs(dw_overfit):
        print(f"  → Backprop shows better generalization")
    else:
        print(f"  → DeltaW shows better generalization")
    
    print("="*60)

    # Generate comparison plots
    print(f"\nGenerating comparison plots...")
    outdir = Path(args.out)
    make_plots(bp_res, alt_res, outdir)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()