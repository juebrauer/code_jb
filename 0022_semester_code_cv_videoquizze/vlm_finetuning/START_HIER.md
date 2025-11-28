# 🤖 Kinematic Chain VLM - Projekt-Übersicht

## 📁 Dateien in diesem Paket

### Hauptdateien

1. **kinematic_chain_vlm.py** (40 KB)
   - Haupt-Skript für das VLM-basierte System
   - Drei Modi: collect, train, test
   - Paligemma Integration
   - GUI-Visualisierung

2. **README_VLM.md** (9.4 KB)
   - Detaillierte Anleitung
   - Installation, Verwendung, Troubleshooting
   - Beispiele und Best Practices

3. **requirements_vlm.txt** (549 Bytes)
   - Alle Python-Abhängigkeiten
   - Direkt installierbar mit pip

4. **quick_start.sh** (4.3 KB)
   - Interaktives Setup-Skript
   - Führt durch den gesamten Workflow
   - Prüft Abhängigkeiten

### Dokumentation

5. **CNN_vs_VLM.md** (4.8 KB)
   - Vergleich der beiden Ansätze
   - Vor- und Nachteile
   - Empfehlungen wann welcher Ansatz

6. **AENDERUNGEN.md** (8.9 KB)
   - Dokumentation der Änderungen vom CNN-Code
   - Für Referenz und Vergleich

### Legacy (für Vergleich)

7. **03_kinematic_chain_il_improved.py** (46 KB)
   - Verbesserte CNN-Version
   - Falls Sie CNN-Ansatz bevorzugen

## 🚀 Quick Start (3 Minuten)

### Methode 1: Automatisch (empfohlen)

```bash
# 1. Skript ausführbar machen
chmod +x quick_start.sh

# 2. Skript starten
./quick_start.sh

# 3. Folgen Sie den Anweisungen im Terminal
```

### Methode 2: Manuell

```bash
# 1. Dependencies installieren
pip install -r requirements_vlm.txt

# 2. Hugging Face Login
huggingface-cli login

# 3. Daten sammeln
python kinematic_chain_vlm.py collect --dof 2 --samples 5000 --output data/dof2

# 4. Modell trainieren
python kinematic_chain_vlm.py train --dof 2 --data data/dof2 --epochs 5 --output models/

# 5. Modell testen
python kinematic_chain_vlm.py test --dof 2 --model models/paligemma_dof2_best
```

## 📊 Was ist neu? (Vergleich zum CNN-Code)

### ✅ Entfernt
- ❌ CNN-Architektur (ActionCNN)
- ❌ CNN-Training Loop
- ❌ CNN-spezifisches Dataset
- ❌ Confusion Matrix, Per-Action Plots

### ✅ Hinzugefügt
- ✅ **VLM Integration** (Paligemma-3B)
- ✅ **Natural Language Actions**
  - Input: Bild + Prompt
  - Output: "Rotate joint 0 clockwise by 1 degree"
- ✅ **LoRA Fine-tuning** (parameter-effizient)
- ✅ **4-bit Quantization** (weniger GPU Memory)
- ✅ **VLM-Dataset Format** (CSV mit Prompt/Answer)
- ✅ **Text Parsing** (VLM Output → Action Index)

### ✅ Gleich geblieben
- ✅ Kinematic Chain Logik
- ✅ Expert Policy (Inverse Kinematics)
- ✅ GUI-Visualisierung
- ✅ Drei Modi (collect, train, test)
- ✅ CSV-basiertes Datenformat

## 🎯 Wichtige Unterschiede

### Datenformat

**CNN:**
```csv
sample_id,image_filename,action
0,image_000000.png,2
```

**VLM:**
```csv
sample_id,image_filename,action,prompt,answer
0,image_000000.png,2,"Analyze the robot arm...","Rotate joint 0 clockwise by 1 degree"
```

### Training

| Aspekt | CNN | VLM |
|--------|-----|-----|
| Dauer | 30-60 Min | 2-4 Stunden |
| GPU Memory | 4-6 GB | 12-16 GB |
| Parameter | ~1-5M | ~3B (mit LoRA ~8-16M trainierbar) |
| Batch Size | 32-64 | 4-8 |
| Epochen | 50-100 | 5-10 |

### Inference

| Aspekt | CNN | VLM |
|--------|-----|-----|
| Latenz | 1-5 ms | 50-200 ms |
| Output | Action Index (0-5) | Text → Parse → Action |
| Interpretierbar | Nein | Ja (natürliche Sprache) |

## 💡 Wann welcher Ansatz?

### Verwende **CNN** wenn:
- ✓ Echtzeit (<10ms) erforderlich
- ✓ Begrenzte GPU (< 8GB)
- ✓ Embedded System
- ✓ Produktion/Deployment
- ✓ Nur ein spezifischer Task

### Verwende **VLM** wenn:
- ✓ Interpretierbarkeit wichtig
- ✓ Multi-Task geplant
- ✓ Natural Language Control gewünscht
- ✓ Forschung/Experimente
- ✓ Bessere Generalisierung nötig
- ✓ GPU mit 12+ GB verfügbar

## 📚 Empfohlene Lese-Reihenfolge

1. **README_VLM.md** - Start hier!
   - Installation
   - Grundlegende Verwendung
   - Troubleshooting

2. **CNN_vs_VLM.md**
   - Verständnis der Unterschiede
   - Entscheidungshilfe

3. **kinematic_chain_vlm.py**
   - Code-Struktur verstehen
   - Bei Bedarf anpassen

4. **AENDERUNGEN.md**
   - Nur falls Sie vom CNN-Code kommen
   - Zeigt was geändert wurde

## 🔧 System-Anforderungen

### Minimum
- Python 3.8+
- 12 GB GPU VRAM (für Training)
- 16 GB RAM
- 20 GB freier Speicherplatz

### Empfohlen
- Python 3.10+
- 24 GB GPU VRAM (RTX 3090/4090)
- 32 GB RAM
- 50 GB freier Speicherplatz (für mehrere Experimente)

### Unterstützte Plattformen
- ✅ Linux (Ubuntu 20.04+)
- ✅ Windows 10/11 mit WSL2
- ⚠️ macOS (nur CPU, sehr langsam)

## 🐛 Häufige Probleme

### "Error loading model"
```bash
# Lösung: Hugging Face Login
huggingface-cli login
```

### "CUDA out of memory"
```bash
# Lösung: Kleinere Batch-Size
python kinematic_chain_vlm.py train --batch-size 2
```

### "CSV file not found"
```bash
# Lösung: Collect-Mode komplett durchlaufen lassen
# Die CSV wird erst am Ende gespeichert!
```

## 📈 Erwartete Ergebnisse

### Nach Datensammlung (5000 Samples)
```
data/dof2_vlm/
├── training_data.csv (5000 rows)
├── metadata.json
└── image_*.png (5000 images, ~500 MB)
```

### Nach Training (5 Epochen)
```
models/
├── paligemma_dof2_best/      # Bestes Modell
│   ├── adapter_config.json
│   ├── adapter_model.safetensors (~30 MB)
│   └── ...
└── paligemma_dof2_final/     # Finales Modell
```

### Im Test-Mode
- **Success Rate**: 85-95%
- **Avg Steps per Scenario**: 50-150 (je nach DOF und Startposition)
- **FPS**: 5-10 (durch VLM-Latenz begrenzt)

## 🎓 Weiterführende Ideen

### Einfache Erweiterungen
1. **Mehr DOF**: `--dof 3` oder `--dof 4`
2. **Mehr Daten**: `--samples 10000`
3. **Längeres Training**: `--epochs 10`

### Fortgeschrittene Erweiterungen
1. **Hindernisse vermeiden**
   - Erweitere Prompt: "Avoid the obstacle (gray circle) and reach target"
   - Zeichne Hindernisse in GUI
   - Sammle neue Daten

2. **Multi-Target**
   - "Reach red target, then blue target, then return to start"
   - Sequential task learning

3. **Constraints**
   - "Keep joint 1 angle below 45 degrees"
   - "Minimize joint 0 movement"

4. **Natural Language Control**
   - User gibt Kommandos in Echtzeit
   - VLM interpretiert und steuert

## 📞 Support

Bei Problemen:
1. Lesen Sie README_VLM.md (Troubleshooting-Sektion)
2. Überprüfen Sie CNN_vs_VLM.md für konzeptuelle Fragen
3. Schauen Sie in den Code (gut kommentiert)

## 🎉 Viel Erfolg!

Das VLM-System bietet spannende Möglichkeiten für:
- Forschung in Vision-Language Models
- Robotik-Anwendungen
- Imitation Learning
- Multi-Modal AI

Viel Spaß beim Experimentieren! 🚀
