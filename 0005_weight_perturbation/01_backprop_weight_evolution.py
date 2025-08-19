import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Stelle sicher, dass das Ausgabeverzeichnis existiert
output_dir = "weight_evolution_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Erzeuge Trainings- und Testdaten
def generate_data(n_samples=1000):
    """
    Erzeugt Daten für eine komplizierte Funktion:
    y = sin(2*x1) + x2^2 - 0.5*cos(3*x3) + x1*x2 + noise
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Erzeuge zufällige Eingabedaten
    X = np.random.uniform(-3, 3, (n_samples, 3))
    
    # Komplizierte Zielfunktion
    y = (np.sin(2 * X[:, 0]) + 
         X[:, 1]**2 - 
         0.5 * np.cos(3 * X[:, 2]) + 
         X[:, 0] * X[:, 1])
    
    # Füge etwas Rauschen hinzu
    noise = np.random.normal(0, 0.1, n_samples)
    y = y + noise
    
    # Konvertiere zu PyTorch Tensoren
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    return X_tensor, y_tensor

# 2. Definiere das MLP
class MLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[64, 32, 16], output_dim=1):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Versteckte Schichten
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Ausgabeschicht
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Speichere Layer-Informationen für bessere Beschriftung
        self.layer_info = {}
        layer_counter = 0
        all_dims = [input_dim] + hidden_dims + [output_dim]
        
        for i, module in enumerate(self.network):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                
                if layer_counter == 0:
                    layer_name = f"Input_to_Hidden1"
                elif layer_counter == len(hidden_dims):
                    layer_name = f"Hidden{layer_counter}_to_Output"
                else:
                    layer_name = f"Hidden{layer_counter}_to_Hidden{layer_counter+1}"
                
                # Speichere die Info mit dem Modulnamen als Schlüssel
                param_prefix = f"network.{i}"
                self.layer_info[param_prefix] = {
                    'name': layer_name,
                    'in_features': in_features,
                    'out_features': out_features,
                    'layer_idx': layer_counter
                }
                layer_counter += 1
    
    def forward(self, x):
        return self.network(x)

# 3. Trainingsschleife mit Gewichtsaufzeichnung
def train_and_record(model, X_train, y_train, X_test, y_test, 
                     n_epochs=500, learning_rate=0.001):
    """
    Trainiert das Modell und zeichnet die Gewichtsentwicklung auf
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dictionary zum Speichern der Gewichtsentwicklung
    weight_history = defaultdict(list)
    bias_history = defaultdict(list)
    loss_history = {'train': [], 'test': []}
    
    # Speichere Layer-Info für spätere Verwendung
    param_to_layer_info = {}
    for name, param in model.named_parameters():
        # Extrahiere den Layer-Präfix (z.B. "network.0" aus "network.0.weight")
        layer_prefix = name.rsplit('.', 1)[0]
        if layer_prefix in model.layer_info:
            param_to_layer_info[name] = model.layer_info[layer_prefix]
    
    print("Starte Training...")
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Berechne Test-Loss
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
        
        # Speichere Losses
        loss_history['train'].append(loss.item())
        loss_history['test'].append(test_loss.item())
        
        # Speichere Gewichte und Biases
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Flatten und als Liste speichern
                weights_flat = param.data.cpu().numpy().flatten()
                for i, w in enumerate(weights_flat):
                    weight_history[f"{name}_w{i}"].append(w)
            elif 'bias' in name:
                biases = param.data.cpu().numpy()
                for i, b in enumerate(biases):
                    bias_history[f"{name}_b{i}"].append(b)
        
        # Fortschrittsanzeige
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                  f"Train Loss: {loss.item():.4f}, "
                  f"Test Loss: {test_loss.item():.4f}")
    
    print("Training abgeschlossen!")
    return weight_history, bias_history, loss_history, param_to_layer_info

# 4. Plotte die Gewichtsentwicklung
def plot_weight_evolution(weight_history, bias_history, loss_history, param_to_layer_info):
    """
    Erstellt Plots für jedes Gewicht und jeden Bias
    """
    print("\nErstelle Plots...")
    
    # Plot für Loss-Entwicklung
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history['train']) + 1)
    plt.plot(epochs, loss_history['train'], 'b-', label='Train Loss', alpha=0.7)
    plt.plot(epochs, loss_history['test'], 'r-', label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training und Test Loss über Zeit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, '00_loss_evolution.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    # Plots für Gewichte
    plot_counter = 1
    
    # Gruppiere Gewichte nach Layer
    layer_weights = defaultdict(list)
    for weight_name in weight_history.keys():
        layer_name = weight_name.split('_w')[0]
        layer_weights[layer_name].append(weight_name)
    
    # Erstelle einen Plot pro Layer mit allen Gewichten
    for layer_name, weights in layer_weights.items():
        n_weights = len(weights)
        
        # Hole Layer-Informationen
        layer_info = param_to_layer_info.get(layer_name, {})
        readable_name = layer_info.get('name', layer_name)
        in_neurons = layer_info.get('in_features', '?')
        out_neurons = layer_info.get('out_features', '?')
        
        # Begrenze die Anzahl der gezeigten Gewichte für bessere Übersichtlichkeit
        max_weights_to_plot = 50
        if n_weights > max_weights_to_plot:
            # Wähle zufällig einige Gewichte aus
            np.random.seed(42)
            selected_indices = np.random.choice(n_weights, max_weights_to_plot, replace=False)
            weights_to_plot = [weights[i] for i in selected_indices]
            plot_info = f" ({max_weights_to_plot} von {n_weights} Gewichten)"
        else:
            weights_to_plot = weights
            plot_info = f" (alle {n_weights} Gewichte)"
        
        plt.figure(figsize=(12, 8))
        
        for weight_name in weights_to_plot:
            history = weight_history[weight_name]
            plt.plot(range(len(history)), history, alpha=0.5, linewidth=0.8)
        
        plt.xlabel('Trainingsschritt')
        plt.ylabel('Gewichtswert')
        plt.title(f'{readable_name} [{in_neurons}→{out_neurons} Neuronen]{plot_info}')
        plt.grid(True, alpha=0.3)
        
        filename = f"{plot_counter:02d}_weights_{readable_name.replace(' ', '_').replace('→', 'to')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')
        plt.close()
        plot_counter += 1
        
        print(f"  Plot erstellt: {filename}")
    
    # Erstelle Plots für Biases pro Layer
    layer_biases = defaultdict(list)
    for bias_name in bias_history.keys():
        layer_name = bias_name.rsplit('_b', 1)[0]
        layer_biases[layer_name].append(bias_name)
    
    for layer_name, biases in layer_biases.items():
        if biases:
            # Hole Layer-Informationen
            layer_info = param_to_layer_info.get(layer_name.replace('.bias', '.weight'), {})
            readable_name = layer_info.get('name', layer_name)
            out_neurons = layer_info.get('out_features', len(biases))
            
            plt.figure(figsize=(10, 6))
            
            for i, bias_name in enumerate(biases):
                history = bias_history[bias_name]
                plt.plot(range(len(history)), history, label=f'Neuron {i}', alpha=0.7)
            
            plt.xlabel('Trainingsschritt')
            plt.ylabel('Bias-Wert')
            plt.title(f'Bias-Entwicklung: {readable_name} [{out_neurons} Neuronen]')
            plt.grid(True, alpha=0.3)
            if len(biases) <= 20:
                plt.legend(ncol=2 if len(biases) > 10 else 1)
            
            filename = f"{plot_counter:02d}_biases_{readable_name.replace(' ', '_').replace('→', 'to')}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')
            plt.close()
            plot_counter += 1
            
            print(f"  Plot erstellt: {filename}")
    
    print(f"\nAlle Plots wurden in '{output_dir}/' gespeichert!")

# Hauptprogramm
def main():
    # Erzeuge Daten
    print("Erzeuge Daten...")
    X, y = generate_data(n_samples=1000)
    
    # Teile in Training und Test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Trainingsdaten: {X_train.shape}")
    print(f"Testdaten: {X_test.shape}")
    
    # Erstelle das Modell
    model = MLP(input_dim=3, hidden_dims=[64, 32, 16], output_dim=1)
    print(f"\nModell-Architektur:")
    print(model)
    
    # Zähle Parameter
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nGesamtzahl der Parameter: {total_params}")
    
    # Trainiere und zeichne auf
    weight_history, bias_history, loss_history, param_to_layer_info = train_and_record(
        model, X_train, y_train, X_test, y_test,
        n_epochs=500, learning_rate=0.001
    )
    
    # Erstelle Plots
    plot_weight_evolution(weight_history, bias_history, loss_history, param_to_layer_info)
    
    # Finale Evaluation
    model.eval()
    with torch.no_grad():
        final_train_pred = model(X_train)
        final_test_pred = model(X_test)
        final_train_loss = nn.MSELoss()(final_train_pred, y_train)
        final_test_loss = nn.MSELoss()(final_test_pred, y_test)
    
    print(f"\nFinale Performance:")
    print(f"  Train Loss: {final_train_loss.item():.4f}")
    print(f"  Test Loss: {final_test_loss.item():.4f}")

if __name__ == "__main__":
    main()