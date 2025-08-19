#!/usr/bin/env python3
"""
Vergleich: Backprop-MLP (PyTorch/Adam) vs. simple DeltaW-Regel (gradientenfrei)

1) Erzeugt Trainings-/Testdaten für eine nichtlineare Ziel-Funktion f(x,y,z)
2) Trainiert ein MLP mit klassischem Backprop (Adam)
3) Trainiert ein identisches MLP mit einer alternativen Regel:
   - Für jeden Parameter wird ein DeltaW gehalten.
   - Pro Mini-Batch wird ein Schritt p := p + DeltaW probiert.
   - Wird der Loss kleiner: Schritt behalten, DeltaW behalten.
   - Wird der Loss größer: Schritt verwerfen und zufälliges neues DeltaW ziehen.
4) Visualisiert den Lernfortschritt (Train-/Test-MSE) und zeigt einen Scatter-Plot
   (Vorhersage vs. Ground Truth) am Ende.

Laufbeispiel:
  python PyTorch_MLP_vs_DeltaW.py --train 5000 --test 1000 --epochs 25 --batch 256 --hidden 64

Hinweis: Für größere Läufe (z.B. 10k+ Samples / 50 Epochen) kann es je nach CPU dauern.
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

# --------------------------- Hilfsfunktionen ---------------------------

def target_function(X: torch.Tensor) -> torch.Tensor:
    """Komplexere nichtlineare Funktion f(x,y,z) mit Cosinus- und Potenzen.
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


def make_dataset(n_samples: int, noise_std: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.empty(n_samples, 3).uniform_(-2.0, 2.0)
    y = target_function(X)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return X, y


def create_mlp(input_dim=3, hidden=64, depth=2) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


def evaluate_mse(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            loss_sum += nn.functional.mse_loss(pred, yb, reduction="sum").item()
            n += yb.numel()
    return loss_sum / n

# --------------------------- Backprop-Training ---------------------------

@dataclass
class TrainResult:
    train_mse: list[float]
    test_mse: list[float]
    model: nn.Module


def train_backprop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   epochs: int = 25, lr: float = 1e-3) -> TrainResult:
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
    return TrainResult(train_hist, test_hist, model)

# --------------------------- Alternative DeltaW-Regel ---------------------------

class DeltaWTrainer:
    def __init__(self, model: nn.Module, base_scale: float = 1e-3):
        self.model = model
        self.base_scale = base_scale
        self.criterion = nn.MSELoss()
        self.delta_ws: list[torch.Tensor] = []
        self._init_deltas()

    def _init_deltas(self):
        self.delta_ws = []
        for p in self.model.parameters():
            p_std = float(p.data.std().item()) if p.data.numel() > 1 else float(p.data.abs().mean().item())
            scale = self.base_scale * (p_std if p_std > 0 else 1.0)
            self.delta_ws.append(torch.randn_like(p.data) * scale)

    def _resample_deltas(self):
        new = []
        for p in self.model.parameters():
            p_std = float(p.data.std().item()) if p.data.numel() > 1 else float(p.data.abs().mean().item())
            scale = self.base_scale * (p_std if p_std > 0 else 1.0)
            new.append(torch.randn_like(p.data) * scale)
        self.delta_ws = new

    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 25):
        train_hist, test_hist = [], []
        for ep in range(epochs):
            self.model.train()
            accepted = 0
            total = 0
            for xb, yb in train_loader:
                with torch.no_grad():
                    loss_old = self.criterion(self.model(xb), yb)
                    # Kandidaten-Update
                    old_params = [p.data.clone() for p in self.model.parameters()]
                    for p, d in zip(self.model.parameters(), self.delta_ws):
                        p.add_(d)
                    loss_new = self.criterion(self.model(xb), yb)
                    total += 1
                    if loss_new.item() < loss_old.item():
                        accepted += 1
                        # Schritt behalten, DeltaW behalten
                    else:
                        # Schritt verwerfen, neue DeltaWs probieren
                        for p, old in zip(self.model.parameters(), old_params):
                            p.data.copy_(old)
                        self._resample_deltas()
            train_hist.append(evaluate_mse(self.model, train_loader))
            test_hist.append(evaluate_mse(self.model, test_loader))
            acc_rate = accepted / max(1, total)
            print(f"[DeltaW] Ep {ep+1:02d}  AccRate={acc_rate:.2f}  TrainMSE={train_hist[-1]:.4f}  TestMSE={test_hist[-1]:.4f}")
        return TrainResult(train_hist, test_hist, self.model)

# --------------------------- Plotting ---------------------------

def make_plots(bp: TrainResult, alt: TrainResult, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(bp.train_mse) + 1)

    # Training Loss
    plt.figure()
    plt.plot(epochs, bp.train_mse, label="Backprop (Train)")
    plt.plot(epochs, alt.train_mse, label="DeltaW-Regel (Train)")
    plt.xlabel("Epoche"); plt.ylabel("MSE"); plt.title("Trainingsfehler über Epochen"); plt.legend()
    plt.savefig(outdir / "train_loss_comparison.png", bbox_inches="tight")

    # Test Loss
    plt.figure()
    plt.plot(epochs, bp.test_mse, label="Backprop (Test)")
    plt.plot(epochs, alt.test_mse, label="DeltaW-Regel (Test)")
    plt.xlabel("Epoche"); plt.ylabel("MSE"); plt.title("Testfehler über Epochen"); plt.legend()
    plt.savefig(outdir / "test_loss_comparison.png", bbox_inches="tight")

    # Scatter: Pred vs True (Test)
    def preds_on(model: nn.Module, X: torch.Tensor) -> np.ndarray:
        model.eval();
        with torch.no_grad():
            return model(X).cpu().numpy().ravel()

    plt.figure()
    # Erzeuge Testdaten erneut mit gleichem Seed, damit beide auf demselben Set ausgewertet werden
    X_test, y_test = make_dataset(2000, noise_std=0.05)
    y_true = y_test.numpy().ravel()
    pred_bp = preds_on(bp.model, X_test)
    pred_alt = preds_on(alt.model, X_test)
    plt.scatter(y_true, pred_bp, s=6, alpha=0.6, label="Backprop")
    plt.scatter(y_true, pred_alt, s=6, alpha=0.6, label="DeltaW-Regel")
    plt.xlabel("Wahr (y)"); plt.ylabel("Vorhersage (ŷ)"); plt.title("Vorhersage vs. Wahr (Testdaten)"); plt.legend()
    plt.savefig(outdir / "pred_vs_true_test.png", bbox_inches="tight")

    print(f"Plots gespeichert in: {outdir.resolve()}")

# --------------------------- Main ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=5000, help="Anzahl Trainingssamples")
    p.add_argument("--test", type=int, default=1000, help="Anzahl Testsamples")
    p.add_argument("--epochs", type=int, default=25, help="Epochen")
    p.add_argument("--batch", type=int, default=256, help="Batchgröße")
    p.add_argument("--hidden", type=int, default=64, help="Hidden-Units pro Layer")
    p.add_argument("--depth", type=int, default=2, help="Anzahl Hidden-Layer")
    p.add_argument("--lr", type=float, default=1e-3, help="Lernrate (Backprop)")
    p.add_argument("--deltaw", type=float, default=1e-3, help="Basisskala für DeltaW-Schritte")
    p.add_argument("--out", type=str, default="outputs", help="Ausgabeverzeichnis für Plots")
    p.add_argument("--seed", type=int, default=0, help="Zufallsseed")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Daten
    X_train, y_train = make_dataset(args.train, noise_std=0.05)
    X_test, y_test = make_dataset(args.test, noise_std=0.05)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch, shuffle=False)

    # Modelle mit identischer Init
    base = create_mlp(hidden=args.hidden, depth=args.depth)
    model_bp = create_mlp(hidden=args.hidden, depth=args.depth)
    model_bp.load_state_dict(base.state_dict())
    model_alt = create_mlp(hidden=args.hidden, depth=args.depth)
    model_alt.load_state_dict(base.state_dict())

    # 2) Backprop
    bp_res = train_backprop(model_bp, train_loader, test_loader, epochs=args.epochs, lr=args.lr)
    print(f"Backprop: letzter Train/Test-MSE: {bp_res.train_mse[-1]:.4f} / {bp_res.test_mse[-1]:.4f}")

    # 3) Alternative DeltaW-Regel
    alt_trainer = DeltaWTrainer(model_alt, base_scale=args.deltaw)
    alt_res = alt_trainer.train(train_loader, test_loader, epochs=args.epochs)
    print(f"DeltaW:  letzter Train/Test-MSE: {alt_res.train_mse[-1]:.4f} / {alt_res.test_mse[-1]:.4f}")

    # 4) Plots
    outdir = Path(args.out)
    make_plots(bp_res, alt_res, outdir)


if __name__ == "__main__":
    main()
