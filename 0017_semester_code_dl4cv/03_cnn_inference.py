# viz_25_samples.py
import argparse
from pathlib import Path
import random
import torch
from torch import nn
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- SimpleCNN (wie im Trainingsskript) -------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data", help="Wurzelordner mit Unterordnern pro Klasse")
    parser.add_argument("--ckpt", type=str, default="best_cnn.pth", help="Pfad zum gespeicherten Checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Zufallssamen")
    parser.add_argument("--num", type=int, default=25, help="Anzahl zufälliger Bilder (zeigt 5x5 Grid)")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint laden (enthält img_size & Klassen)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]

    # Transforms: fürs Inferenz-Input (wie Validation)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    infer_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Fürs Anzeigen (ohne Norm, gleiche Größe)
    display_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])

    # Dataset (liest Dateipfade & Targets)
    ds = datasets.ImageFolder(root=args.root)

    # Modell wiederherstellen
    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # 25 (oder weniger) zufällige Indizes ziehen
    n = min(args.num, len(ds))
    indices = random.sample(range(len(ds)), n)

    # Vorbereiten der Figur 5x5
    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()

    # Durch zufällige Bilder iterieren
    with torch.no_grad():
        for i, idx in enumerate(indices):
            path, true_label = ds.samples[idx]  # (Pfad, Klassenindex)
            # Bild fürs Modell
            img_in = infer_tfms(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            logits = model(img_in)
            pred_idx = logits.argmax(1).item()
            pred_name = classes[pred_idx]

            # Bild fürs Anzeigen
            disp_img = display_tfms(Image.open(path).convert("RGB"))
            ax = axes[i]
            ax.imshow(disp_img)
            # Titel mit Vorhersage (und wahrem Label)
            title = f"Pred: {pred_name}\nTrue: {classes[true_label]}"
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        # Leere Achsen ausblenden, falls < 25 Bilder
        for j in range(i+1, rows*cols):
            axes[j].axis("off")

    fig.suptitle("25 zufällige Vorhersagen (5x5)", fontsize=14)
    plt.tight_layout()
    out_path = Path("predictions_5x5.png")
    plt.savefig(out_path, dpi=150)
    print(f"Gespeichert unter: {out_path.resolve()}")
    plt.show()

main()
