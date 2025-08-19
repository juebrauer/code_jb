#!/usr/bin/env python3
"""
Comparison: Backprop-MLP (PyTorch/Adam) vs. PSO (Particle Swarm Optimization)

1) Generates training/test data for a nonlinear target function f(x,y,z)
2) Trains an MLP with classical backprop (Adam)
3) Trains an identical MLP with PSO:
   - Maintains a swarm of particles (parameter configurations)
   - Each particle has position, velocity, personal best, and global best
   - Updates follow PSO dynamics: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
4) Visualizes learning progress (Train/Test MSE) and shows scatter plot
   (Prediction vs. Ground Truth) at the end.

Example run:
  python PyTorch_MLP_vs_PSO.py --train 5000 --test 1000 --epochs 25 --batch 256 --hidden 64

Note: For larger runs (e.g., 10k+ samples / 50 epochs) it may take time depending on CPU.
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
from typing import List, Tuple

# --------------------------- Helper Functions ---------------------------

def target_function(X: torch.Tensor) -> torch.Tensor:
    """Complex nonlinear function f(x,y,z) with cosine and powers.
    X: (N,3) => (x, y, z)
    return: (N,1)
    """
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    f = (
        torch.cos(1.5 * x)
        + 0.5 * (y ** 2)
        - 0.3 * (z ** 3)
        + 0.1 * x * y * z
        + 0.2 * torch.cos(y * z)
        + 0.05 * (x ** 2) * z
    )
    return f.unsqueeze(1)


def make_dataset(n_samples: int, noise_std: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic dataset with nonlinear target function."""
    X = torch.empty(n_samples, 3).uniform_(-2.0, 2.0)
    y = target_function(X)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return X, y


def create_mlp(input_dim=3, hidden=64, depth=2) -> nn.Module:
    """Create a simple MLP with tanh activations."""
    layers: List[nn.Module] = [nn.Linear(input_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


def evaluate_mse(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate mean squared error on a data loader."""
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            loss_sum += nn.functional.mse_loss(pred, yb, reduction="sum").item()
            n += yb.numel()
    return loss_sum / n


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    params = []
    for p in model.parameters():
        params.append(p.data.view(-1))
    return torch.cat(params)


def unflatten_parameters(model: nn.Module, flat_params: torch.Tensor):
    """Load flattened parameters back into model."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = flat_params[idx:idx+numel].view_as(p.data)
        idx += numel

# --------------------------- Backprop Training ---------------------------

@dataclass
class TrainResult:
    train_mse: List[float]
    test_mse: List[float]
    model: nn.Module


def train_backprop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   epochs: int = 25, lr: float = 1e-3) -> TrainResult:
    """Train model using standard backpropagation with Adam optimizer."""
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_hist, test_hist = [], []
    
    for ep in range(epochs):
        model.train()
        run = 0.0
        cnt = 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            run += loss.item() * yb.size(0)
            cnt += yb.size(0)
        train_hist.append(run / cnt)
        test_hist.append(evaluate_mse(model, test_loader))
        print(f"[Backprop] Epoch {ep+1:02d}  TrainMSE={train_hist[-1]:.4f}  TestMSE={test_hist[-1]:.4f}")
    
    return TrainResult(train_hist, test_hist, model)

# --------------------------- PSO (Particle Swarm Optimization) ---------------------------

class Particle:
    """Represents a single particle in the swarm."""
    
    def __init__(self, dim: int, bounds: Tuple[float, float] = (-2.0, 2.0)):
        """Initialize particle with random position and velocity."""
        self.position = torch.FloatTensor(dim).uniform_(*bounds)
        self.velocity = torch.FloatTensor(dim).uniform_(-abs(bounds[1]-bounds[0])*0.1, 
                                                        abs(bounds[1]-bounds[0])*0.1)
        self.best_position = self.position.clone()
        self.best_fitness = float('inf')
        self.fitness = float('inf')


class PSOTrainer:
    """PSO trainer for neural network optimization."""
    
    def __init__(self, model: nn.Module, 
                 n_particles: int = 20,
                 w: float = 0.7,  # Inertia weight
                 c1: float = 1.5,  # Cognitive parameter
                 c2: float = 1.5,  # Social parameter
                 bounds: Tuple[float, float] = (-2.0, 2.0)):
        """
        Initialize PSO trainer.
        
        Args:
            model: Neural network model to optimize
            n_particles: Number of particles in swarm
            w: Inertia weight (momentum)
            c1: Cognitive parameter (attraction to personal best)
            c2: Social parameter (attraction to global best)
            bounds: Parameter bounds for initialization
        """
        self.model = model
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.criterion = nn.MSELoss()
        
        # Get dimension of parameter space
        self.dim = sum(p.numel() for p in model.parameters())
        
        # Initialize swarm
        self.particles = [Particle(self.dim, bounds) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
    def evaluate_fitness(self, particle: Particle, train_loader: DataLoader) -> float:
        """Evaluate fitness of a particle on training data."""
        # Load particle position into model
        unflatten_parameters(self.model, particle.position)
        
        # Calculate total loss on training data
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                total_loss += loss.item() * yb.size(0)
                n_samples += yb.size(0)
        
        return total_loss / n_samples
    
    def update_velocity_position(self, particle: Particle):
        """Update particle velocity and position using PSO equations."""
        # Random factors for stochastic behavior
        r1 = torch.rand(self.dim)
        r2 = torch.rand(self.dim)
        
        # Update velocity: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        particle.velocity = self.w * particle.velocity + cognitive + social
        
        # Update position
        particle.position = particle.position + particle.velocity
        
        # Apply bounds (optional, can be removed for unbounded optimization)
        particle.position = torch.clamp(particle.position, self.bounds[0], self.bounds[1])
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 25) -> TrainResult:
        """Train model using PSO."""
        train_hist, test_hist = [], []
        
        # Initialize with current model parameters as one particle
        initial_params = flatten_parameters(self.model)
        self.particles[0].position = initial_params
        
        for epoch in range(epochs):
            # Evaluate all particles
            for particle in self.particles:
                particle.fitness = self.evaluate_fitness(particle, train_loader)
                
                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.clone()
                
                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.clone()
            
            # Update velocities and positions
            for particle in self.particles:
                self.update_velocity_position(particle)
            
            # Load best solution into model for evaluation
            unflatten_parameters(self.model, self.global_best_position)
            
            # Record metrics
            train_mse = evaluate_mse(self.model, train_loader)
            test_mse = evaluate_mse(self.model, test_loader)
            train_hist.append(train_mse)
            test_hist.append(test_mse)
            
            # Print progress
            print(f"[PSO] Epoch {epoch+1:02d}  BestFitness={self.global_best_fitness:.4f}  "
                  f"TrainMSE={train_mse:.4f}  TestMSE={test_mse:.4f}")
            
            # Optional: Adaptive inertia weight decay
            self.w = max(0.4, self.w * 0.98)
        
        # Ensure best solution is loaded into model
        unflatten_parameters(self.model, self.global_best_position)
        
        return TrainResult(train_hist, test_hist, self.model)

# --------------------------- Plotting ---------------------------

def make_plots(bp: TrainResult, pso: TrainResult, outdir: Path):
    """Generate comparison plots between Backprop and PSO."""
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(bp.train_mse) + 1)

    # Training Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, bp.train_mse, 'b-', label="Backprop (Train)", linewidth=2)
    plt.plot(epochs, pso.train_mse, 'r-', label="PSO (Train)", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Error over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Test Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, bp.test_mse, 'b--', label="Backprop (Test)", linewidth=2)
    plt.plot(epochs, pso.test_mse, 'r--', label="PSO (Test)", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Test Error over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "loss_comparison.png", dpi=150, bbox_inches="tight")

    # Scatter: Predictions vs True (Test)
    def preds_on(model: nn.Module, X: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(X).cpu().numpy().ravel()

    plt.figure(figsize=(10, 5))
    
    # Generate test data with same seed for fair comparison
    X_test, y_test = make_dataset(2000, noise_std=0.05)
    y_true = y_test.numpy().ravel()
    
    # Backprop predictions
    plt.subplot(1, 2, 1)
    pred_bp = preds_on(bp.model, X_test)
    plt.scatter(y_true, pred_bp, s=10, alpha=0.6, c='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.5)
    plt.xlabel("True y")
    plt.ylabel("Predicted ŷ")
    plt.title(f"Backprop (MSE: {bp.test_mse[-1]:.4f})")
    plt.grid(True, alpha=0.3)
    
    # PSO predictions
    plt.subplot(1, 2, 2)
    pred_pso = preds_on(pso.model, X_test)
    plt.scatter(y_true, pred_pso, s=10, alpha=0.6, c='red')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.5)
    plt.xlabel("True y")
    plt.ylabel("Predicted ŷ")
    plt.title(f"PSO (MSE: {pso.test_mse[-1]:.4f})")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "pred_vs_true_test.png", dpi=150, bbox_inches="tight")

    print(f"Plots saved in: {outdir.resolve()}")

# --------------------------- Main ---------------------------

def main():
    p = argparse.ArgumentParser(description="Compare Backprop vs PSO for MLP training")
    p.add_argument("--train", type=int, default=5000, help="Number of training samples")
    p.add_argument("--test", type=int, default=1000, help="Number of test samples")
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    p.add_argument("--batch", type=int, default=256, help="Batch size")
    p.add_argument("--hidden", type=int, default=64, help="Hidden units per layer")
    p.add_argument("--depth", type=int, default=2, help="Number of hidden layers")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Backprop)")
    p.add_argument("--particles", type=int, default=50, help="Number of particles in PSO")
    p.add_argument("--w", type=float, default=0.7, help="PSO inertia weight")
    p.add_argument("--c1", type=float, default=1.5, help="PSO cognitive parameter")
    p.add_argument("--c2", type=float, default=1.5, help="PSO social parameter")
    p.add_argument("--out", type=str, default="outputs", help="Output directory for plots")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    args = p.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Generate data
    print("Generating synthetic data...")
    X_train, y_train = make_dataset(args.train, noise_std=0.05)
    X_test, y_test = make_dataset(args.test, noise_std=0.05)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch, shuffle=False)

    # Create models with identical initialization
    print("Creating models...")
    base = create_mlp(hidden=args.hidden, depth=args.depth)
    model_bp = create_mlp(hidden=args.hidden, depth=args.depth)
    model_bp.load_state_dict(base.state_dict())
    model_pso = create_mlp(hidden=args.hidden, depth=args.depth)
    model_pso.load_state_dict(base.state_dict())

    # Train with Backprop
    print("\n" + "="*50)
    print("Training with Backpropagation (Adam)...")
    print("="*50)
    bp_res = train_backprop(model_bp, train_loader, test_loader, epochs=args.epochs, lr=args.lr)
    print(f"\nBackprop Final: Train MSE={bp_res.train_mse[-1]:.4f}, Test MSE={bp_res.test_mse[-1]:.4f}")

    # Train with PSO
    print("\n" + "="*50)
    print("Training with Particle Swarm Optimization...")
    print("="*50)
    pso_trainer = PSOTrainer(model_pso, n_particles=args.particles, 
                            w=args.w, c1=args.c1, c2=args.c2)
    pso_res = pso_trainer.train(train_loader, test_loader, epochs=args.epochs)
    print(f"\nPSO Final: Train MSE={pso_res.train_mse[-1]:.4f}, Test MSE={pso_res.test_mse[-1]:.4f}")

    # Generate plots
    print("\n" + "="*50)
    print("Generating comparison plots...")
    print("="*50)
    outdir = Path(args.out)
    make_plots(bp_res, pso_res, outdir)
    
    # Summary comparison
    print("\n" + "="*50)
    print("SUMMARY COMPARISON")
    print("="*50)
    print(f"{'Method':<15} {'Final Train MSE':<20} {'Final Test MSE':<20}")
    print("-"*55)
    print(f"{'Backprop':<15} {bp_res.train_mse[-1]:<20.4f} {bp_res.test_mse[-1]:<20.4f}")
    print(f"{'PSO':<15} {pso_res.train_mse[-1]:<20.4f} {pso_res.test_mse[-1]:<20.4f}")


if __name__ == "__main__":
    main()