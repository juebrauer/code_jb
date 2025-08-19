# -*- coding: utf-8 -*-
"""
Vergleich von Trainingsmethoden für ein Neuronales Netz:
1. Klassische Backpropagation mit Adam-Optimizer.
2. Eine alternative Methode basierend auf einer zufälligen Suche ("DeltaW"-Regel).

KORRIGIERTE VERSION: Beide Modelle starten nun garantiert mit den exakt gleichen
zufälligen Gewichten, um einen fairen Vergleich zu ermöglichen.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy

# --- 1. Datengenerierung ---

def complex_function(x, y, z):
    """Definiert eine nicht-lineare Funktion als Lernziel."""
    return np.cos(2 * np.pi * x) + y**2 + np.sin(z**3)

# Globale Einstellungen für Reproduzierbarkeit
np.random.seed(42)
torch.manual_seed(42)

# Generiere die Beispieldaten
n_samples = 1000
X = np.random.rand(n_samples, 3) * 2 - 1  # Werte zwischen -1 und 1
y = complex_function(X[:, 0], X[:, 1], X[:, 2]).reshape(-1, 1)

# Aufteilung in Trainings- und Testdaten (80/20)
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Konvertiere NumPy-Arrays zu PyTorch Tensoren
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

print("--- Datengenerierung abgeschlossen ---")
print(f"Trainingsdaten Shape: {X_train_t.shape}")
print(f"Testdaten Shape: {X_test_t.shape}\n")


# --- 2. Modellarchitektur und gemeinsamer Startpunkt ---

class MLP(nn.Module):
    """Ein einfaches Multilayer Perceptron (MLP)."""
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Initialisiere die beiden Modelle
model_bp = MLP()
model_alt = MLP()

# KORREKTUR: Speichere den zufälligen Anfangszustand von model_bp.
# Dies geschieht VOR jeglichem Training.
initial_state = copy.deepcopy(model_bp.state_dict())

# Lade diesen exakten Anfangszustand in das zweite Modell.
# Jetzt ist ein fairer Vergleich sichergestellt.
model_alt.load_state_dict(initial_state)
print("--- Modelle initialisiert und auf den gleichen Startpunkt gesetzt ---\n")


# --- 3. Training mit Backpropagation ---

# Definiere Loss-Funktion und Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model_bp.parameters(), lr=0.001)

# Trainings-Loop für Backpropagation
print("--- Starte Training mit Backpropagation ---")
epochs = 1000
bp_losses = []

for epoch in range(epochs):
    model_bp.train()
    y_pred = model_bp(X_train_t)
    loss = loss_fn(y_pred, y_train_t)
    bp_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Backprop Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
print("--- Training mit Backpropagation abgeschlossen ---\n")


# --- 4. Training mit der alternativen "DeltaW"-Regel ---

# Hyperparameter für die DeltaW-Methode
learning_rate_alt = 0.001
n_steps = epochs * 10 

# Initialisiere die DeltaWs (zufällige Richtungen) für jedes Gewicht
delta_ws = [torch.randn_like(p) * learning_rate_alt for p in model_alt.parameters()]

# Berechne den initialen Loss als Referenzpunkt
with torch.no_grad():
    last_loss = loss_fn(model_alt(X_train_t), y_train_t)

alt_losses = [last_loss.item()]

print("--- Starte Training mit alternativer 'DeltaW'-Regel ---")
# Trainings-Loop für die alternative Methode
for step in range(n_steps):
    current_weights = [p.clone() for p in model_alt.parameters()]
    
    with torch.no_grad():
        for i, p in enumerate(model_alt.parameters()):
            p.data += delta_ws[i]
    
    with torch.no_grad():
        y_pred = model_alt(X_train_t)
        new_loss = loss_fn(y_pred, y_train_t)

    if new_loss < last_loss:
        last_loss = new_loss
    else:
        with torch.no_grad():
            for i, p in enumerate(model_alt.parameters()):
                p.data = current_weights[i]
        delta_ws = [torch.randn_like(p) * learning_rate_alt for p in model_alt.parameters()]

    alt_losses.append(last_loss.item())

    if (step + 1) % (20 * 10) == 0:
        print(f'Alternative Step [{step+1}/{n_steps}], Loss: {last_loss.item():.4f}')
print("--- Training mit alternativer Regel abgeschlossen ---\n")


# --- 5. Vergleich, Plotting und finale Auswertung ---

print("--- Erstelle Vergleichsplot ---")
plt.figure(figsize=(12, 7))

# Plot für Backpropagation
plt.plot(bp_losses, label='Backpropagation (Adam)', color='blue', linewidth=2)

# Plot für die alternative Methode
sample_indices = np.linspace(0, len(alt_losses) - 1, len(bp_losses), dtype=int)
sampled_alt_losses = [alt_losses[i] for i in sample_indices]
plt.plot(sampled_alt_losses, label='Alternative "DeltaW"-Regel', color='orange', linestyle='--', marker='o', markersize=4)

# Plot-Einstellungen
plt.xlabel('Epochen')
plt.ylabel('Mean Squared Error (Loss)')
plt.title('Lernfortschritt: Backpropagation vs. DeltaW-Regel')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.yscale('log')
plt.tight_layout()
plt.show()

# Finale Evaluation auf den ungesehenen Testdaten
model_bp.eval()
model_alt.eval()
with torch.no_grad():
    test_pred_bp = model_bp(X_test_t)
    test_loss_bp = loss_fn(test_pred_bp, y_test_t)
    
    test_pred_alt = model_alt(X_test_t)
    test_loss_alt = loss_fn(test_pred_alt, y_test_t)

print("\n--- Finale Auswertung auf Testdaten ---")
print(f"Test-Loss (Backpropagation): {test_loss_bp.item():.4f}")
print(f"Test-Loss (Alternative Methode): {test_loss_alt.item():.4f}")
