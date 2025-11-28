#!/bin/bash
# Quick Start Script für Kinematic Chain VLM
# Dieses Skript führt Sie durch den gesamten Workflow

set -e  # Exit on error

echo "=================================="
echo "Kinematic Chain VLM - Quick Start"
echo "=================================="
echo ""

# Farben für bessere Lesbarkeit
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funktion für farbige Ausgabe
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parameter
DOF=${DOF:-2}
SAMPLES=${SAMPLES:-5000}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-4}

echo "Konfiguration:"
echo "  DOF: $DOF"
echo "  Samples: $SAMPLES"
echo "  Epochen: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Schritt 1: Abhängigkeiten prüfen
print_step "1/5 - Überprüfe Abhängigkeiten..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 ist nicht installiert!"
    exit 1
fi

print_info "Python 3 gefunden: $(python3 --version)"

# Schritt 2: Virtual Environment (optional)
read -p "Möchten Sie ein Virtual Environment erstellen? (empfohlen) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "2/5 - Erstelle Virtual Environment..."
    python3 -m venv venv
    source venv/bin/activate
    print_info "Virtual Environment aktiviert"
else
    print_info "Überspringe Virtual Environment"
fi

# Schritt 3: Pakete installieren
print_step "3/5 - Installiere Pakete..."
print_info "Dies kann einige Minuten dauern..."

if [ -f "requirements_vlm.txt" ]; then
    pip install -q -r requirements_vlm.txt
    print_info "Alle Pakete installiert"
else
    print_error "requirements_vlm.txt nicht gefunden!"
    exit 1
fi

# Schritt 4: Hugging Face Login
print_step "4/5 - Hugging Face Setup..."
print_info "Sie benötigen einen Hugging Face Account und müssen die Paligemma Lizenz akzeptieren."
print_info "1. Erstellen Sie einen Account: https://huggingface.co/join"
print_info "2. Akzeptieren Sie die Lizenz: https://huggingface.co/google/paligemma-3b-pt-224"
print_info "3. Generieren Sie ein Token: https://huggingface.co/settings/tokens"
echo ""

read -p "Haben Sie die Lizenz akzeptiert und ein Token? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starten Sie den Login-Prozess..."
    huggingface-cli login
else
    print_error "Bitte akzeptieren Sie zuerst die Lizenz!"
    exit 1
fi

# Schritt 5: Workflow-Auswahl
print_step "5/5 - Workflow starten..."
echo ""
echo "Wählen Sie einen Modus:"
echo "  1) Nur Daten sammeln (Collect)"
echo "  2) Nur Training (benötigt existierende Daten)"
echo "  3) Nur Testen (benötigt trainiertes Modell)"
echo "  4) Vollständiger Workflow (Collect → Train → Test)"
echo ""
read -p "Ihre Wahl [1-4]: " -n 1 -r
echo

case $REPLY in
    1)
        print_step "Starte Daten-Sammlung..."
        python3 kinematic_chain_vlm.py collect \
            --dof $DOF \
            --samples $SAMPLES \
            --output "data/dof${DOF}_vlm"
        
        print_info "Daten gesammelt in: data/dof${DOF}_vlm"
        ;;
    
    2)
        read -p "Pfad zu den Trainingsdaten: " DATA_PATH
        print_step "Starte Training..."
        python3 kinematic_chain_vlm.py train \
            --dof $DOF \
            --data "$DATA_PATH" \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --output "models/"
        
        print_info "Modell trainiert und gespeichert in: models/"
        ;;
    
    3)
        read -p "Pfad zum Modell: " MODEL_PATH
        print_step "Starte Test-Modus..."
        python3 kinematic_chain_vlm.py test \
            --dof $DOF \
            --model "$MODEL_PATH" \
            --verbose
        ;;
    
    4)
        print_step "Starte vollständigen Workflow..."
        
        # Collect
        print_info "Phase 1/3: Daten sammeln..."
        python3 kinematic_chain_vlm.py collect \
            --dof $DOF \
            --samples $SAMPLES \
            --output "data/dof${DOF}_vlm"
        
        print_info "✓ Daten gesammelt!"
        echo ""
        
        # Train
        print_info "Phase 2/3: Modell trainieren..."
        print_info "Dies kann 2-4 Stunden dauern..."
        python3 kinematic_chain_vlm.py train \
            --dof $DOF \
            --data "data/dof${DOF}_vlm" \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --output "models/"
        
        print_info "✓ Modell trainiert!"
        echo ""
        
        # Test
        print_info "Phase 3/3: Modell testen..."
        python3 kinematic_chain_vlm.py test \
            --dof $DOF \
            --model "models/paligemma_dof${DOF}_best" \
            --verbose
        
        print_info "✓ Vollständiger Workflow abgeschlossen!"
        ;;
    
    *)
        print_error "Ungültige Auswahl!"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Fertig!"
echo "=================================="
