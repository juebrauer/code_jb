import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import csv
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

class RobotCNN(nn.Module):
    """CNN for robot action prediction from 4 camera views - Joint-Space only"""
    
    def __init__(self, action_dim=8, image_size=224):
        super(RobotCNN, self).__init__()
        self.image_size = image_size
        self.action_dim = action_dim
        
        # CNN for the combined 4 images (12 channels for 4 RGB images)
        self.features = nn.Sequential(
            # First layer
            nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second layer
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth layer
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Adaptive pooling for uniform output size
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Smaller Fully Connected Layers for 8 outputs
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.action_dim)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class RobotDataset(Dataset):
    """Dataset for robot training data with Joint-Space actions only (8 dimensions)"""
    
    def __init__(self, data_dir, transform=None, image_size=224, max_episodes=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Find all episode directories
        self.episodes = []
        for episode_dir in sorted(self.data_dir.glob("episode_*")):
            if episode_dir.is_dir():
                # Check if actions.csv and all camera directories exist
                actions_file = episode_dir / "actions.csv"
                camera_dirs = {
                    'corner': episode_dir / "corner",
                    'front': episode_dir / "front", 
                    'side': episode_dir / "side",
                    'top': episode_dir / "top"
                }
                
                if actions_file.exists() and all(cam_dir.exists() for cam_dir in camera_dirs.values()):
                    self.episodes.append(episode_dir)
                    
                    # Limit number of episodes if specified
                    if max_episodes and len(self.episodes) >= max_episodes:
                        break
        
        print(f"Found: {len(self.episodes)} valid episodes (max_episodes: {max_episodes})")
        
        # Load all data into memory
        self.data = []
        self.load_all_data()
    
    def load_all_data(self):
        """Load all episode data with Joint-Space actions only"""
        total_frames = 0
        
        for episode_dir in self.episodes:
            episode_name = episode_dir.name
            print(f"Loading episode: {episode_name}")
            
            # Load actions.csv
            actions_file = episode_dir / "actions.csv"
            with open(actions_file, 'r') as f:
                reader = csv.DictReader(f)
                actions_data = list(reader)
            
            # For each frame in the episode
            valid_frames = 0
            for row in actions_data:
                frame = int(row['frame'])
                
                # Paths to the 4 camera images
                image_paths = {
                    'corner': episode_dir / row['corner_image'],
                    'front': episode_dir / row['front_image'],
                    'side': episode_dir / row['side_image'],
                    'top': episode_dir / row['top_image']
                }
                
                # Check if all images exist
                if all(path.exists() for path in image_paths.values()):
                    # Extract action vector from CSV (8 dimensions: 7 joints + gripper)
                    # 
                    # IMPROVED ACTION VECTOR STRUCTURE:
                    # Only Joint-Space control for direct robot actuation
                    action = [
                        # JOINT ANGLES (7 values in radians)
                        # Franka Panda robot arm has 7 degrees of freedom
                        float(row['q0']),       # Shoulder pan (rotation around Z-axis)
                        float(row['q1']),       # Shoulder lift (arm raising/lowering)
                        float(row['q2']),       # Arm twist (upper arm rotation)
                        float(row['q3']),       # Elbow flex (elbow bending)
                        float(row['q4']),       # Wrist twist (wrist rotation)
                        float(row['q5']),       # Wrist bend (wrist bending)
                        float(row['q6']),       # Hand twist (final hand rotation)
                        
                        # GRIPPER CONTROL (1 value in meters)
                        # Controls the opening width of the parallel gripper
                        float(row['gripper_open_width'])  # 0.0=closed, 0.08=fully open
                    ]
                    
                    # Convert to numpy array with proper clipping
                    action = np.array(action, dtype=np.float32)
                    
                    # Clipping for safety and stability
                    action[:7] = np.clip(action[:7], -2.8, 2.8)  # Joint limits
                    action[7] = np.clip(action[7], 0.0, 0.08)    # Gripper limits
                    
                    # Debug: Check action vector length on first sample
                    if len(self.data) == 0:
                        print(f"  Action vector length: {len(action)} (expected: 8)")
                        print(f"  Joint values: {action[:7]}")
                        print(f"  Gripper: {action[7]}")
                        if len(action) != 8:
                            print(f"  WARNING: Action vector has {len(action)} dimensions instead of 8!")
                    
                    self.data.append({
                        'images': image_paths,
                        'action': action,
                        'episode': episode_name,
                        'frame': frame
                    })
                    valid_frames += 1
                else:
                    # Debug: Show missing images
                    missing = [view for view, path in image_paths.items() if not path.exists()]
                    print(f"  Frame {frame}: Missing images: {missing}")
            
            print(f"  Valid frames: {valid_frames}")
            total_frames += valid_frames
        
        print(f"Total training data: {len(self.data)} frames from {len(self.episodes)} episodes")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load and process the 4 images
        images = []
        for view in ['corner', 'front', 'side', 'top']:
            try:
                img = Image.open(sample['images'][view]).convert('RGB')
                img = img.resize((self.image_size, self.image_size))
                
                if self.transform:
                    img = self.transform(img)
                else:
                    # Standard transformation
                    img = transforms.ToTensor()(img)
                
                images.append(img)
            except Exception as e:
                print(f"Error loading {sample['images'][view]}: {e}")
                # Fallback: Black image
                img = torch.zeros(3, self.image_size, self.image_size)
                images.append(img)
        
        # Concatenate all 4 images into one tensor (concatenate channels)
        combined_image = torch.cat(images, dim=0)  # Shape: [12, H, W]
        
        action = torch.tensor(sample['action'], dtype=torch.float32)
        
        return combined_image, action

def analyze_dataset(data_dir, max_episodes=None):
    """Analyze the data structure and show statistics for Joint-Space actions"""
    data_path = Path(data_dir)
    
    print("="*60)
    print("DATASET ANALYSIS - JOINT-SPACE ACTIONS")
    print("="*60)
    
    episodes = sorted(data_path.glob("episode_*"))
    if max_episodes:
        episodes = episodes[:max_episodes]
    
    print(f"Analyzing episodes: {len(episodes)} (max_episodes: {max_episodes})")
    
    total_frames = 0
    action_samples = []
    
    for episode_dir in episodes[:3]:  # Analyze only first 3 episodes for speed
        print(f"\nEpisode: {episode_dir.name}")
        
        actions_file = episode_dir / "actions.csv"
        if actions_file.exists():
            with open(actions_file, 'r') as f:
                reader = csv.DictReader(f)
                actions_data = list(reader)
                print(f"  Frames: {len(actions_data)}")
                total_frames += len(actions_data)
                
                # Collect action samples
                for i, row in enumerate(actions_data):
                    if i % 10 == 0 and len(action_samples) < 50:
                        if len(action_samples) == 0:
                            print(f"  CSV columns: {list(row.keys())}")
                            # Count actual action columns
                            action_cols = [col for col in row.keys() if col not in ['frame', 'top_image', 'side_image', 'front_image', 'corner_image']]
                            print(f"  Action columns: {action_cols}")
                            print(f"  Number of action columns: {len(action_cols)}")
                        
                        # NEW ACTION VECTOR: Only joints + gripper (8 dimensions)
                        action = [
                            # Joint angles (7 values)
                            float(row['q0']), float(row['q1']), float(row['q2']), float(row['q3']),
                            float(row['q4']), float(row['q5']), float(row['q6']),
                            # Gripper control (1 value)
                            float(row['gripper_open_width'])
                        ]
                        
                        # Debug: Verify action length and print first action breakdown
                        if len(action_samples) == 0:
                            print(f"  Action vector length: {len(action)} (should be 8)")
                            print(f"  First action breakdown:")
                            print(f"    joints: {action[:7]} (7 vals)")
                            print(f"    gripper: {action[7]} (1 val)")
                        
                        action_samples.append(action)
        
        # Check camera directories
        for cam in ['corner', 'front', 'side', 'top']:
            cam_dir = episode_dir / cam
            if cam_dir.exists():
                images = list(cam_dir.glob("*.png"))
                print(f"  {cam}: {len(images)} images")
    
    print(f"\nEstimated total frames (all episodes): ~{total_frames * len(episodes) // min(3, len(episodes))}")
    
    # Action vector statistics
    if action_samples:
        actions_array = np.array(action_samples)
        print(f"\nAction vector statistics from {len(action_samples)} samples:")
        print(f"Action vector dimensions: {actions_array.shape[1]}")
        print("Action vector ranges:")
        labels = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'gripper_open_width']
        
        for i, label in enumerate(labels[:actions_array.shape[1]]):
            values = actions_array[:, i]
            range_str = f"{values.min():.3f} to {values.max():.3f}"
            std_str = f"std: {values.std():.3f}"
            print(f"  {label:20}: {range_str:20} ({std_str})")
        
        # Check if any dimension has zero variance (potential issue)
        zero_variance_dims = []
        for i in range(actions_array.shape[1]):
            if actions_array[:, i].std() < 1e-6:
                zero_variance_dims.append(labels[i] if i < len(labels) else f"dim_{i}")
        
        if zero_variance_dims:
            print(f"\nWARNING: Dimensions with near-zero variance: {zero_variance_dims}")
            print("These dimensions might not contribute to learning!")
    else:
        print("\nNo action samples collected for analysis.")
    
    print("="*60)

def weighted_mse_loss(predictions, targets):
    """MSE loss with higher weight for gripper actions"""
    joint_loss = nn.functional.mse_loss(predictions[:, :7], targets[:, :7])
    gripper_loss = nn.functional.mse_loss(predictions[:, 7:8], targets[:, 7:8])
    return joint_loss + 2.0 * gripper_loss  # Gripper is 2x more important

def train_model(data_dir, model_save_dir="model_checkpoints", 
                num_epochs=100, batch_size=16, learning_rate=0.0001,
                image_size=224, action_dim=8, max_episodes=None):
    """Train the improved CNN model with Joint-Space actions"""
    
    # Create directory for model checkpoints
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Device setup with fallback for older GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check CUDA compatibility
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
        
        # Check if GPU is compatible
        try:
            test_tensor = torch.randn(1, 1, device=device)
            _ = test_tensor + 1
            print(f"Training on: {device}")
        except Exception as e:
            print(f"CUDA detected but not compatible: {e}")
            print("Falling back to CPU training...")
            device = torch.device('cpu')
    else:
        print(f"Training on: {device}")
    
    print(f"Final device: {device}")
    
    # Improved data transforms - minimal augmentation for robotics
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Very light augmentation - only brightness
        transforms.ColorJitter(brightness=0.05, contrast=0.0, saturation=0.0, hue=0.0),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = RobotDataset(data_dir, transform=None, image_size=image_size, max_episodes=max_episodes)
    
    if len(full_dataset) == 0:
        raise ValueError("No valid data found! Please check the data structure.")
    
    # Train/Validation split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Create separate datasets with different transforms
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    full_dataset.transform = train_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training data: {len(train_dataset)}")
    print(f"Validation data: {len(val_dataset)}")
    
    # Create model
    model = RobotCNN(action_dim=action_dim, image_size=image_size)
    model = model.to(device)
    
    # Loss and optimizer with improved settings
    criterion = weighted_mse_loss  # Use weighted loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        full_dataset.transform = train_transform
        
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, actions) in enumerate(train_loader):
            images, actions = images.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')
        
        # Validation
        model.eval()
        full_dataset.transform = val_transform
        
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for images, actions in val_loader:
                images, actions = images.to(device), actions.to(device)
                outputs = model(images)
                loss = criterion(outputs, actions)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
        
        # Calculate average losses
        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss / val_samples
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduler
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training Loss: {avg_train_loss:.6f}')
        print(f'  Validation Loss: {avg_val_loss:.6f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
        
        # Save model after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'action_dim': action_dim,
            'image_size': image_size
        }
        
        checkpoint_path = os.path.join(model_save_dir, f'robot_cnn_joints_epoch_{epoch+1:03d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_save_dir, 'robot_cnn_joints_best.pth')
            torch.save(checkpoint, best_model_path)
            print(f'New best model saved: {best_model_path}')
        
        print()
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Validation Loss Detail')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'training_progress_joints.png'))
    plt.show()
    
    print("Training completed!")
    return model, train_losses, val_losses

def load_model(checkpoint_path, device='cpu'):
    """Load a saved model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = RobotCNN(
        action_dim=checkpoint['action_dim'],
        image_size=checkpoint['image_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Training Loss: {checkpoint['train_loss']:.6f}")
    print(f"Validation Loss: {checkpoint['val_loss']:.6f}")
    
    return model, checkpoint

def test_model_on_episode(model_path, episode_dir, device='cpu'):
    """Test the model on a specific episode"""
    print(f"Testing model on episode: {episode_dir}")
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    model.eval()
    
    # Load episode
    dataset = RobotDataset(Path(episode_dir).parent, image_size=checkpoint['image_size'])
    episode_name = Path(episode_dir).name
    
    # Filter data for this episode
    episode_data = [item for item in dataset.data if item['episode'] == episode_name]
    
    if not episode_data:
        print(f"No data found for episode {episode_name}!")
        return
    
    print(f"Testing on {len(episode_data)} frames...")
    
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for i, sample in enumerate(episode_data[:100:10]):  # Test every 10th frame up to 100
            # Load images
            images = []
            for view in ['corner', 'front', 'side', 'top']:
                img = Image.open(sample['images'][view]).convert('RGB')
                img = img.resize((checkpoint['image_size'], checkpoint['image_size']))
                img = transforms.ToTensor()(img)
                images.append(img)
            
            combined_image = torch.cat(images, dim=0).unsqueeze(0).to(device)
            true_action = torch.tensor(sample['action']).unsqueeze(0).to(device)
            
            # Prediction
            predicted_action = model(combined_image)
            loss = criterion(predicted_action, true_action)
            total_loss += loss.item()
            
            print(f"Frame {sample['frame']:3d}: Loss = {loss.item():.6f}")
            
            # Show comparison for first frame
            if i == 0:
                print("Comparison (joints + gripper):")
                pred = predicted_action[0].cpu().numpy()
                true = true_action[0].cpu().numpy()
                joint_labels = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'gripper']
                for j in range(len(pred)):
                    label = joint_labels[j] if j < len(joint_labels) else f"dim_{j}"
                    print(f"  {label}: Pred={pred[j]:.3f}, True={true[j]:.3f}, Diff={abs(pred[j]-true[j]):.3f}")
    
    avg_loss = total_loss / min(10, len(episode_data))
    print(f"Average loss: {avg_loss:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Improved Robot CNN Training and Testing (Joint-Space)')
    parser.add_argument('--mode', choices=['train', 'test'], required=True,
                        help='Mode: train or test')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing episodes')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum number of episodes to use (default: all)')
    parser.add_argument('--model_dir', type=str, default='model_checkpoints',
                        help='Directory to save/load model checkpoints')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Specific model path for testing')
    parser.add_argument('--test_episode', type=str, default=None,
                        help='Specific episode directory for testing')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--action_dim', type=int, default=8,
                        help='Action vector dimensions (8 for joints+gripper)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found!")
        return 1
    
    # Analyze dataset first
    print("Analyzing dataset...")
    analyze_dataset(args.input_dir, args.max_episodes)
    
    if args.mode == 'train':
        print(f"\nStarting training with {args.max_episodes or 'all'} episodes...")
        print(f"Action dimensions: {args.action_dim} (Joint-Space)")
        try:
            model, train_losses, val_losses = train_model(
                data_dir=args.input_dir,
                model_save_dir=args.model_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                image_size=args.image_size,
                action_dim=args.action_dim,
                max_episodes=args.max_episodes
            )
            
            print(f"\nTraining completed successfully!")
            print(f"Final training loss: {train_losses[-1]:.6f}")
            print(f"Final validation loss: {val_losses[-1]:.6f}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.mode == 'test':
        # Determine model path
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(args.model_dir, "robot_cnn_joints_best.pth")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            return 1
        
        # Determine episode to test
        if args.test_episode:
            test_episode_dir = args.test_episode
        else:
            # Use first episode in input directory
            episodes = sorted(Path(args.input_dir).glob("episode_*"))
            if not episodes:
                print(f"Error: No episodes found in {args.input_dir}")
                return 1
            test_episode_dir = str(episodes[0])
        
        if not os.path.exists(test_episode_dir):
            print(f"Error: Test episode directory {test_episode_dir} not found!")
            return 1
        
        print(f"\nTesting model {model_path} on episode {test_episode_dir}...")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            test_model_on_episode(model_path, test_episode_dir, device)
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())