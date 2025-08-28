import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import time

class ForwardForwardLayer:
    """
    A single layer that learns using the Forward-Forward algorithm.
    Each layer learns to distinguish between positive and negative data
    by maximizing the goodness for positive data and minimizing it for negative data.
    """
    
    def __init__(self, input_dim, output_dim, threshold=2.0, learning_rate=0.03):
        """
        Initialize a Forward-Forward layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of neurons in this layer
            threshold: Threshold for goodness function
            learning_rate: Learning rate for weight updates
        """
        # Initialize weights with small random values
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.zeros(output_dim)
        self.threshold = threshold
        self.learning_rate = learning_rate
        
    def forward(self, x):
        """
        Forward pass through the layer.
        
        Args:
            x: Input data (batch_size, input_dim)
            
        Returns:
            activations: Layer activations after ReLU
        """
        # Linear transformation followed by ReLU activation
        z = np.dot(x, self.weights) + self.bias
        return np.maximum(0, z)  # ReLU activation
    
    def compute_goodness(self, activations):
        """
        Compute the goodness of the layer's activations.
        Goodness is defined as the sum of squared activations per sample.
        
        Args:
            activations: Layer activations (batch_size, output_dim)
            
        Returns:
            goodness: Goodness value for each sample (batch_size,)
        """
        return np.sum(activations ** 2, axis=1)
    
    def train_step(self, x_pos, x_neg):
        """
        Perform one training step using positive and negative data.
        
        Args:
            x_pos: Positive (real) data
            x_neg: Negative (corrupted) data
        """
        # Forward pass for positive data
        act_pos = self.forward(x_pos)
        goodness_pos = self.compute_goodness(act_pos)  # Shape: (batch_size,)
        
        # Forward pass for negative data
        act_neg = self.forward(x_neg)
        goodness_neg = self.compute_goodness(act_neg)  # Shape: (batch_size,)
        
        # Compute gradients properly
        # For positive data: we want goodness > threshold
        pos_loss = np.maximum(0, self.threshold - goodness_pos)
        
        # For negative data: we want goodness < threshold  
        neg_loss = np.maximum(0, goodness_neg - self.threshold)
        
        # Compute gradients w.r.t. activations
        grad_pos = np.zeros_like(act_pos)
        grad_neg = np.zeros_like(act_neg)
        
        # Only update where ReLU is active and loss is positive
        for i in range(len(x_pos)):
            if pos_loss[i] > 0:
                grad_pos[i] = 2 * act_pos[i] * (act_pos[i] > 0)
        
        for i in range(len(x_neg)):
            if neg_loss[i] > 0:
                grad_neg[i] = -2 * act_neg[i] * (act_neg[i] > 0)
        
        # Update weights and biases
        if len(x_pos) > 0:
            self.weights += self.learning_rate * np.dot(x_pos.T, grad_pos) / len(x_pos)
            self.bias += self.learning_rate * np.mean(grad_pos, axis=0)
            
        if len(x_neg) > 0:
            self.weights += self.learning_rate * np.dot(x_neg.T, grad_neg) / len(x_neg)
            self.bias += self.learning_rate * np.mean(grad_neg, axis=0)
        
        return np.mean(goodness_pos), np.mean(goodness_neg)


class ForwardForwardNetwork:
    """
    A multi-layer network trained with the Forward-Forward algorithm.
    """
    
    def __init__(self, layer_dims, threshold=2.0, learning_rate=0.03):
        """
        Initialize the Forward-Forward network.
        
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ...]
            threshold: Threshold for goodness function
            learning_rate: Learning rate for all layers
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                ForwardForwardLayer(
                    layer_dims[i], 
                    layer_dims[i+1], 
                    threshold, 
                    learning_rate
                )
            )
    
    def forward(self, x):
        """
        Forward pass through the entire network.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def create_negative_data(self, x, y, method='random_labels'):
        """
        Create negative (corrupted) data for training.
        """
        if method == 'random_labels':
            n_classes = y.shape[1]
            wrong_labels = np.zeros_like(y)
            for i in range(len(y)):
                true_class = np.argmax(y[i])
                wrong_class = np.random.choice([c for c in range(n_classes) if c != true_class])
                wrong_labels[i, wrong_class] = 1.0
            
            x_with_wrong_labels = np.concatenate([x, wrong_labels], axis=1)
            return x_with_wrong_labels
            
        elif method == 'mix_samples':
            idx = np.random.permutation(len(x))
            mixed_x = (x + x[idx]) / 2
            return np.concatenate([mixed_x, y], axis=1)
            
        elif method == 'noise':
            noise = np.random.randn(*x.shape) * 0.3
            noisy_x = np.clip(x + noise, 0, 1)
            return np.concatenate([noisy_x, y], axis=1)
        else:
            wrong_labels = np.roll(y, np.random.randint(1, y.shape[1]), axis=0)
            return np.concatenate([x, wrong_labels], axis=1)
    
    def train(self, x_train, y_train, epochs=10, batch_size=64, verbose=True):
        """
        Train the network using the Forward-Forward algorithm.
        """
        n_samples = len(x_train)
        losses = {'positive': [], 'negative': []}
        
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        print(f"Total batches per epoch: {n_samples // batch_size}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Shuffle data
            idx = np.random.permutation(n_samples)
            x_train_shuffled = x_train[idx]
            y_train_shuffled = y_train[idx]
            
            epoch_pos_goodness = []
            epoch_neg_goodness = []
            
            # Train in batches
            for i in range(0, n_samples, batch_size):
                x_batch = x_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Combine input with label information for positive data
                x_pos = np.concatenate([x_batch, y_batch], axis=1)
                
                # Create negative data with different methods for variety
                methods = ['random_labels', 'mix_samples', 'noise']
                method = np.random.choice(methods)
                x_neg = self.create_negative_data(x_batch, y_batch, method=method)
                
                # Train each layer independently
                current_pos = x_pos
                current_neg = x_neg
                
                for layer_idx, layer in enumerate(self.layers):
                    pos_goodness, neg_goodness = layer.train_step(current_pos, current_neg)
                    epoch_pos_goodness.append(pos_goodness)
                    epoch_neg_goodness.append(neg_goodness)
                    
                    if layer_idx < len(self.layers) - 1:
                        current_pos = layer.forward(current_pos)
                        current_neg = layer.forward(current_neg)
            
            # Record average goodness for this epoch
            avg_pos = np.mean(epoch_pos_goodness) if epoch_pos_goodness else 0
            avg_neg = np.mean(epoch_neg_goodness) if epoch_neg_goodness else 0
            
            losses['positive'].append(avg_pos)
            losses['negative'].append(avg_neg)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{epochs} - "
                      f"Pos: {avg_pos:.3f}, Neg: {avg_neg:.3f}, "
                      f"Diff: {avg_pos - avg_neg:.3f}, "
                      f"Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        
        return losses
    
    def predict(self, x, y_all_classes, verbose=False):
        """
        Make predictions by finding which label gives highest goodness.
        """
        predictions = []
        
        if verbose:
            print("Making predictions...")
        
        for i, sample in enumerate(x):
            if verbose and i % 10 == 0:
                print(f"  Predicting sample {i+1}/{len(x)}")
                
            goodness_scores = []
            
            # Try each possible label
            for class_idx, label in enumerate(y_all_classes):
                x_with_label = np.concatenate([sample, label]).reshape(1, -1)
                
                # Forward pass through all layers and compute total goodness
                current = x_with_label
                total_goodness = 0
                
                for layer in self.layers:
                    current = layer.forward(current)
                    layer_goodness = np.sum(current ** 2)
                    total_goodness += layer_goodness
                
                goodness_scores.append(total_goodness)
            
            best_class = np.argmax(goodness_scores)
            predictions.append(best_class)
        
        return np.array(predictions)


def load_mnist_data(n_samples=1000, test_size=0.2, random_state=42):
    """
    Load and preprocess MNIST data.
    """
    print(f"Loading MNIST data (n_samples={n_samples})...")
    
    # Load MNIST
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = np.array(mnist.data.astype('float32')) / 255.0
    y = np.array(mnist.target.astype('int'))
    
    # Use subset for faster demo
    X = X[:n_samples]
    y = y[:n_samples]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Convert labels to one-hot encoding
    n_classes = 10
    y_train_onehot = np.eye(n_classes)[y_train]
    y_test_onehot = np.eye(n_classes)[y_test]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test


def visualize_results(losses, y_test, predictions, show_plot=True):
    """
    Visualize training progress and results.
    """
    if not show_plot:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training curves
    ax1.plot(losses['positive'], label='Positive Data Goodness', color='green', linewidth=2)
    ax1.plot(losses['negative'], label='Negative Data Goodness', color='red', linewidth=2)
    ax1.axhline(y=1.0, color='gray', linestyle='--', label='Threshold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Goodness')
    ax1.set_title('Training Progress: Forward-Forward Algorithm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    im = ax2.imshow(cm, cmap='Blues')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Prediction Results')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Forward-Forward Algorithm on MNIST',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of MNIST samples to use')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Goodness threshold')
    
    # Architecture parameters
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden layer dimensions (e.g. --hidden-dims 512 256 128)')
    
    # Evaluation parameters
    parser.add_argument('--eval-samples', type=int, default=50,
                        help='Number of test samples to evaluate on')
    
    # Output parameters
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output during prediction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("FORWARD-FORWARD ALGORITHM ON MNIST")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Samples: {args.samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Eval samples: {args.eval_samples}")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test = load_mnist_data(
        n_samples=args.samples, 
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Create network architecture
    input_dim = X_train.shape[1] + y_train_onehot.shape[1]  # 784 + 10
    layer_dims = [input_dim] + args.hidden_dims
    
    print(f"\nNetwork architecture: {' -> '.join(map(str, layer_dims))}")
    
    network = ForwardForwardNetwork(
        layer_dims=layer_dims,
        threshold=args.threshold,
        learning_rate=args.lr
    )
    
    # Train network
    print(f"\nTraining network...")
    losses = network.train(
        X_train, y_train_onehot, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        verbose=True
    )
    
    # Make predictions
    print(f"\nEvaluating on {args.eval_samples} test samples...")
    all_classes = np.eye(10)
    predictions = network.predict(
        X_test[:args.eval_samples], 
        all_classes,
        verbose=args.verbose
    )
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test[:args.eval_samples])
    
    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Test accuracy: {accuracy:.1%} ({accuracy*args.eval_samples:.0f}/{args.eval_samples})")
    print(f"Final positive goodness: {losses['positive'][-1]:.3f}")
    print(f"Final negative goodness: {losses['negative'][-1]:.3f}")
    print(f"Goodness separation: {losses['positive'][-1] - losses['negative'][-1]:.3f}")
    
    # Show example predictions
    print(f"\nExample predictions (first 10 samples):")
    for i in range(min(10, args.eval_samples)):
        status = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"  {status} True: {y_test[i]}, Predicted: {predictions[i]}")
    
    # Visualize results
    if not args.no_plot:
        visualize_results(losses, y_test[:args.eval_samples], predictions, show_plot=True)
    
    print("\n" + "=" * 60)
    print("Training completed!")