import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Ensure the output directory exists for saving plots
output_dir = "weight_evolution_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Generate training and test data
def generate_data(n_samples=1000):
    """
    Generates synthetic data for a complex non-linear function:
    y = sin(2*x1) + x2^2 - 0.5*cos(3*x3) + x1*x2 + noise
    
    This function creates a challenging regression problem that requires
    a neural network to learn multiple non-linear relationships between
    input features and the target variable.
    
    Args:
        n_samples (int): Number of data samples to generate
    
    Returns:
        X_tensor: Input features as a PyTorch tensor of shape (n_samples, 3)
        y_tensor: Target values as a PyTorch tensor of shape (n_samples, 1)
    """
    # Set seeds for reproducibility across numpy and PyTorch
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate random input data uniformly distributed between -3 and 3
    # This range ensures good coverage of the non-linear functions used
    X = np.random.uniform(-3, 3, (n_samples, 3))
    
    # Define the complex target function combining multiple non-linearities:
    # - sin(2*x1): Periodic component from first feature
    # - x2^2: Quadratic relationship with second feature
    # - 0.5*cos(3*x3): Another periodic component from third feature
    # - x1*x2: Interaction term between first two features
    y = (np.sin(2 * X[:, 0]) + 
         X[:, 1]**2 - 
         0.5 * np.cos(3 * X[:, 2]) + 
         X[:, 0] * X[:, 1])
    
    # Add Gaussian noise to make the problem more realistic
    # Small noise (std=0.1) ensures the signal is still learnable
    noise = np.random.normal(0, 0.1, n_samples)
    y = y + noise
    
    # Convert numpy arrays to PyTorch tensors for neural network training
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)  # Reshape to column vector
    
    return X_tensor, y_tensor

# 2. Define the Multi-Layer Perceptron (MLP) architecture
class MLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[64, 32, 16], output_dim=1):
        """
        Multi-Layer Perceptron with configurable architecture.
        
        This class creates a fully connected neural network with ReLU activations
        between layers. The architecture progressively reduces dimensionality
        from input to output, creating a funnel-like structure that helps
        learn hierarchical representations.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Number of output features
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with ReLU activations
        # Each hidden layer consists of a linear transformation followed by ReLU
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # Non-linearity crucial for learning complex functions
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression tasks)
        # This allows the network to output any real value
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)
        
        # Store layer information for better labeling in plots
        # This metadata helps create more interpretable visualizations
        self.layer_info = {}
        layer_counter = 0
        all_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Iterate through the network to catalog each linear layer
        for i, module in enumerate(self.network):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                
                # Create human-readable layer names based on position
                if layer_counter == 0:
                    layer_name = f"Input_to_Hidden1"
                elif layer_counter == len(hidden_dims):
                    layer_name = f"Hidden{layer_counter}_to_Output"
                else:
                    layer_name = f"Hidden{layer_counter}_to_Hidden{layer_counter+1}"
                
                # Store the info with the module index as key
                # This allows us to map parameter names to layer descriptions
                param_prefix = f"network.{i}"
                self.layer_info[param_prefix] = {
                    'name': layer_name,
                    'in_features': in_features,
                    'out_features': out_features,
                    'layer_idx': layer_counter
                }
                layer_counter += 1
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

# 3. Training loop with weight recording
def train_and_record(model, X_train, y_train, X_test, y_test, 
                     n_epochs=500, learning_rate=0.001):
    """
    Trains the model while recording the evolution of all weights and biases.
    
    This function performs standard gradient descent training while maintaining
    a complete history of how each parameter changes over time. This allows
    us to visualize the learning dynamics and understand how backpropagation
    adjusts weights to minimize the loss.
    
    Args:
        model: The neural network model to train
        X_train: Training input features
        y_train: Training target values
        X_test: Test input features
        y_test: Test target values
        n_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
    
    Returns:
        weight_history: Dictionary containing the history of all weights
        bias_history: Dictionary containing the history of all biases
        loss_history: Dictionary with train and test loss histories
        param_to_layer_info: Mapping from parameter names to layer information
    """
    # Mean Squared Error loss for regression
    criterion = nn.MSELoss()
    
    # Adam optimizer: adaptive learning rates with momentum
    # More sophisticated than basic SGD, helps with convergence
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dictionaries to store the evolution of parameters over time
    weight_history = defaultdict(list)  # Stores weight values at each epoch
    bias_history = defaultdict(list)      # Stores bias values at each epoch
    loss_history = {'train': [], 'test': []}  # Tracks loss progression
    
    # Create mapping from parameter names to layer information
    # This helps us create meaningful labels in visualizations
    param_to_layer_info = {}
    for name, param in model.named_parameters():
        # Extract the layer prefix (e.g., "network.0" from "network.0.weight")
        layer_prefix = name.rsplit('.', 1)[0]
        if layer_prefix in model.layer_info:
            param_to_layer_info[name] = model.layer_info[layer_prefix]
    
    print("Starting training...")
    
    # Main training loop
    for epoch in range(n_epochs):
        # Set model to training mode (enables dropout, batch norm, etc. if present)
        model.train()
        
        # Zero out gradients from previous iteration
        # PyTorch accumulates gradients by default, so we need to clear them
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(X_train)
        
        # Compute loss between predictions and true values
        loss = criterion(outputs, y_train)
        
        # Backward pass: compute gradients via automatic differentiation
        # This is where backpropagation happens!
        loss.backward()
        
        # Update weights based on gradients
        # The optimizer applies the update rule (e.g., w = w - lr * gradient)
        optimizer.step()
        
        # Evaluate on test set (without computing gradients)
        model.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient computation for efficiency
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
        
        # Record losses for plotting learning curves
        loss_history['train'].append(loss.item())
        loss_history['test'].append(test_loss.item())
        
        # Record all weights and biases at this epoch
        # This allows us to visualize how each parameter evolves
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Flatten weight matrix and save each element separately
                # This gives us fine-grained tracking of individual connections
                weights_flat = param.data.cpu().numpy().flatten()
                for i, w in enumerate(weights_flat):
                    weight_history[f"{name}_w{i}"].append(w)
            elif 'bias' in name:
                # Save each bias term separately
                biases = param.data.cpu().numpy()
                for i, b in enumerate(biases):
                    bias_history[f"{name}_b{i}"].append(b)
        
        # Progress reporting every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                  f"Train Loss: {loss.item():.4f}, "
                  f"Test Loss: {test_loss.item():.4f}")
    
    print("Training completed!")
    return weight_history, bias_history, loss_history, param_to_layer_info

# 4. Plot the weight evolution
def plot_weight_evolution(weight_history, bias_history, loss_history, param_to_layer_info):
    """
    Creates visualization plots showing how weights and biases evolve during training.
    
    This function generates multiple plots:
    1. Loss curves showing training and test loss over epochs
    2. Weight evolution plots for each layer showing how individual weights change
    3. Bias evolution plots for each layer
    
    The visualizations help understand:
    - Whether the model is overfitting (diverging train/test loss)
    - Which layers are most active during learning
    - Whether weights are converging or still changing
    - If there are any unusual patterns in weight updates
    
    Args:
        weight_history: Dictionary with weight values over time
        bias_history: Dictionary with bias values over time
        loss_history: Dictionary with train/test losses over time
        param_to_layer_info: Mapping of parameter names to layer information
    """
    print("\nCreating plots...")
    
    # Plot 1: Loss evolution over training
    # This shows the overall learning progress and helps identify overfitting
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history['train']) + 1)
    plt.plot(epochs, loss_history['train'], 'b-', label='Train Loss', alpha=0.7)
    plt.plot(epochs, loss_history['test'], 'r-', label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, '00_loss_evolution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    plot_counter = 1
    
    # Group weights by layer for organized plotting
    # This creates one plot per layer rather than per weight
    layer_weights = defaultdict(list)
    for weight_name in weight_history.keys():
        layer_name = weight_name.split('_w')[0]
        layer_weights[layer_name].append(weight_name)
    
    # Create one plot per layer showing all weights in that layer
    for layer_name, weights in layer_weights.items():
        n_weights = len(weights)
        
        # Get layer information for meaningful labels
        layer_info = param_to_layer_info.get(layer_name, {})
        readable_name = layer_info.get('name', layer_name)
        in_neurons = layer_info.get('in_features', '?')
        out_neurons = layer_info.get('out_features', '?')
        
        # Limit the number of weights plotted for clarity
        # Large layers might have thousands of weights, making plots unreadable
        max_weights_to_plot = 50
        if n_weights > max_weights_to_plot:
            # Randomly sample a subset of weights for visualization
            np.random.seed(42)  # Consistent sampling across runs
            selected_indices = np.random.choice(n_weights, max_weights_to_plot, replace=False)
            weights_to_plot = [weights[i] for i in selected_indices]
            plot_info = f" ({max_weights_to_plot} of {n_weights} weights)"
        else:
            weights_to_plot = weights
            plot_info = f" (all {n_weights} weights)"
        
        # Create the weight evolution plot for this layer
        plt.figure(figsize=(12, 8))
        
        # Plot each weight's trajectory over training
        for weight_name in weights_to_plot:
            history = weight_history[weight_name]
            plt.plot(range(len(history)), history, alpha=0.5, linewidth=0.8)
        
        plt.xlabel('Training Step')
        plt.ylabel('Weight Value')
        plt.title(f'{readable_name} [{in_neurons}→{out_neurons} neurons]{plot_info}')
        plt.grid(True, alpha=0.3)
        
        # Save with descriptive filename
        filename = f"{plot_counter:02d}_weights_{readable_name.replace(' ', '_').replace('→', 'to')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')
        plt.close()
        plot_counter += 1
        
        print(f"  Plot created: {filename}")
    
    # Create plots for biases per layer
    # Biases often show different learning dynamics than weights
    layer_biases = defaultdict(list)
    for bias_name in bias_history.keys():
        layer_name = bias_name.rsplit('_b', 1)[0]
        layer_biases[layer_name].append(bias_name)
    
    for layer_name, biases in layer_biases.items():
        if biases:
            # Get layer information for labels
            layer_info = param_to_layer_info.get(layer_name.replace('.bias', '.weight'), {})
            readable_name = layer_info.get('name', layer_name)
            out_neurons = layer_info.get('out_features', len(biases))
            
            plt.figure(figsize=(10, 6))
            
            # Plot each bias term's evolution
            for i, bias_name in enumerate(biases):
                history = bias_history[bias_name]
                plt.plot(range(len(history)), history, label=f'Neuron {i}', alpha=0.7)
            
            plt.xlabel('Training Step')
            plt.ylabel('Bias Value')
            plt.title(f'Bias Evolution: {readable_name} [{out_neurons} neurons]')
            plt.grid(True, alpha=0.3)
            
            # Add legend if not too many biases
            if len(biases) <= 20:
                plt.legend(ncol=2 if len(biases) > 10 else 1)
            
            # Save bias evolution plot
            filename = f"{plot_counter:02d}_biases_{readable_name.replace(' ', '_').replace('→', 'to')}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')
            plt.close()
            plot_counter += 1
            
            print(f"  Plot created: {filename}")
    
    print(f"\nAll plots saved in '{output_dir}/'!")

# Main program execution
def main():
    """
    Main execution function that orchestrates the entire experiment:
    1. Data generation
    2. Model creation
    3. Training with weight recording
    4. Visualization of weight evolution
    5. Final performance evaluation
    
    This demonstrates how neural networks learn through backpropagation
    by visualizing the actual weight changes during training.
    """
    # Generate synthetic dataset
    print("Generating data...")
    X, y = generate_data(n_samples=1000)
    
    # Split into training (80%) and test (20%) sets
    # This allows us to monitor generalization during training
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create the model with specified architecture
    # Architecture: 3 inputs -> 64 hidden -> 32 hidden -> 16 hidden -> 1 output
    # This creates a progressively narrowing network that learns hierarchical features
    model = MLP(input_dim=3, hidden_dims=[64, 32, 16], output_dim=1)
    print(f"\nModel Architecture:")
    print(model)
    
    # Count total trainable parameters
    # This gives us an idea of the model's capacity
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")
    
    # Train the model while recording weight evolution
    weight_history, bias_history, loss_history, param_to_layer_info = train_and_record(
        model, X_train, y_train, X_test, y_test,
        n_epochs=500, learning_rate=0.001
    )
    
    # Create visualization plots
    plot_weight_evolution(weight_history, bias_history, loss_history, param_to_layer_info)
    
    # Final evaluation to report model performance
    model.eval()
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_test_pred = model(X_test)
        final_train_loss = nn.MSELoss()(final_train_pred, y_train)
        final_test_loss = nn.MSELoss()(final_test_pred, y_test)
    
    print(f"\nFinal Performance:")
    print(f"  Train Loss: {final_train_loss.item():.4f}")
    print(f"  Test Loss: {final_test_loss.item():.4f}")

# Entry point of the script
if __name__ == "__main__":
    main()