import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Setze Seed für Reproduzierbarkeit
np.random.seed(42)

print("=" * 80)
print("QLoRA Quantisierung Demo - Schritt für Schritt")
print("=" * 80)

# ============================================================================
# Schritt 1: Generiere simulierte Modellgewichte
# ============================================================================
print("\n📊 SCHRITT 1: Generiere simulierte Modellgewichte")
print("-" * 80)

# Erstelle 256 Gewichte mit unterschiedlichen Wertebereichen
# um realistische Bedingungen zu simulieren
N = 1024
N_half = N // 2
weights_part1 = np.random.normal(0.5, 0.1, N_half)  # Bereich ~0.2-0.8
weights_part2 = np.random.normal(2.0, 0.5, N_half)  # Bereich ~1.0-3.0

original_weights = np.concatenate([weights_part1, weights_part2])

print(f"Anzahl Gewichte: {len(original_weights)}")
print(f"Wertebereich: [{original_weights.min():.4f}, {original_weights.max():.4f}]")
print(f"Originalgröße (FP32): {original_weights.nbytes} Bytes")
print(f"Erste 10 Gewichte: {original_weights[:10]}")

# ============================================================================
# Schritt 2: Teile in Blöcke auf
# ============================================================================
print("\n🔲 SCHRITT 2: Teile Gewichte in Blöcke auf")
print("-" * 80)

BLOCK_SIZE = 512
num_blocks = len(original_weights) // BLOCK_SIZE

print(f"Blockgröße: {BLOCK_SIZE}")
print(f"Anzahl Blöcke: {num_blocks}")

# Teile die Gewichte in Blöcke
blocks = []
for i in range(num_blocks):
    start_idx = i * BLOCK_SIZE
    end_idx = start_idx + BLOCK_SIZE
    block = original_weights[start_idx:end_idx]
    blocks.append(block)
    
    print(f"\nBlock {i+1}:")
    print(f"  Wertebereich: [{block.min():.4f}, {block.max():.4f}]")
    print(f"  Mittelwert: {block.mean():.4f}")
    print(f"  Std.abweichung: {block.std():.4f}")

# ============================================================================
# Schritt 3: 4-Bit Quantisierung (NF4)
# ============================================================================
print("\n🔢 SCHRITT 3: 4-Bit Quantisierung (NormalFloat)")
print("-" * 80)

def quantize_block_to_4bit(block):
    """
    Quantisiert einen Block zu 4-Bit unter Verwendung von NF4-ähnlicher Logik
    """
    # Berechne Skalierungsfaktor (absmax)
    absmax = np.abs(block).max()
    scale = absmax / 7.0  # 4-Bit signed: -7 bis +7
    
    # Normalisiere und quantisiere
    normalized = block / scale
    quantized = np.round(np.clip(normalized, -7, 7)).astype(np.int8)
    
    return quantized, scale

# Quantisiere jeden Block
quantized_blocks = []
scales_fp32 = []

quantized_weights_size2 = 0
for i, block in enumerate(blocks):
    q_block, scale = quantize_block_to_4bit(block)
    quantized_blocks.append(q_block)
    scales_fp32.append(scale)
    
    print(f"\nBlock {i+1} Quantisierung:")
    print(f"  Scale (FP32): {scale:.6f}")
    print(f"  Quantisierte Werte (4-Bit): {q_block[:10]}")
    print(f"  Wertebereich quantisiert: [{q_block.min()}, {q_block.max()}]")

    quantized_weights_size2 += q_block.nbytes

# Berechne Speicher nach erster Quantisierung
quantized_weights_size = len(original_weights) * 0.5  # 4 Bit = 0.5 Bytes

scales_fp32_size = num_blocks * 4  # Jeder Scale ist 32-Bit = 4 Bytes
total_size_after_q1 = quantized_weights_size + scales_fp32_size

print(f"\n💾 Speichernutzung nach 4-Bit Quantisierung:")
print(f"  Quantisierte Gewichte (4-Bit): {quantized_weights_size} Bytes")
print(f"  Quantisierte Gewichte2 (4-Bit): {quantized_weights_size2} Bytes")
print(f"  Scales (FP32): {scales_fp32_size} Bytes")
print(f"  TOTAL: {total_size_after_q1} Bytes")
print(f"  Kompression: {original_weights.nbytes / total_size_after_q1:.2f}x")

# ============================================================================
# Schritt 4: Double Quantization
# ============================================================================
print("\n🔢🔢 SCHRITT 4: Double Quantization der Scales")
print("-" * 80)

def quantize_scales_to_8bit(scales):
    """
    Quantisiert die Skalierungsfaktoren selbst zu 8-Bit
    """
    scales_array = np.array(scales)
    
    # Globaler Skalierungsfaktor für alle Scales
    absmax_scales = np.abs(scales_array).max()
    scale_of_scales = absmax_scales / 127.0  # 8-Bit signed: -127 bis +127
    
    # Quantisiere die Scales
    normalized_scales = scales_array / scale_of_scales
    quantized_scales = np.round(np.clip(normalized_scales, -127, 127)).astype(np.int8)
    
    return quantized_scales, scale_of_scales

quantized_scales, scale_of_scales = quantize_scales_to_8bit(scales_fp32)

print(f"Scale der Scales (FP32): {scale_of_scales:.8f}")
print(f"\nOriginale Scales (FP32):")
for i, scale in enumerate(scales_fp32):
    print(f"  Block {i+1}: {scale:.6f}")

print(f"\nQuantisierte Scales (8-Bit):")
for i, q_scale in enumerate(quantized_scales):
    print(f"  Block {i+1}: {q_scale}")

# Berechne Speicher nach Double Quantization
scales_8bit_size = num_blocks * 1  # Jeder Scale ist 8-Bit = 1 Byte
scale_of_scales_size = 4  # Ein FP32 Wert = 4 Bytes
total_size_after_dq = quantized_weights_size + scales_8bit_size + scale_of_scales_size

print(f"\n💾 Speichernutzung nach Double Quantization:")
print(f"  Quantisierte Gewichte (4-Bit): {quantized_weights_size} Bytes")
print(f"  Quantisierte Scales (8-Bit): {scales_8bit_size} Bytes")
print(f"  Scale der Scales (FP32): {scale_of_scales_size} Bytes")
print(f"  TOTAL: {total_size_after_dq} Bytes")
print(f"  Einsparung durch DQ: {total_size_after_q1 - total_size_after_dq} Bytes")
print(f"  Gesamtkompression: {original_weights.nbytes / total_size_after_dq:.2f}x")

# ============================================================================
# Schritt 5: Dequantisierung
# ============================================================================
print("\n🔓 SCHRITT 5: Dequantisierung (Rekonstruktion)")
print("-" * 80)

def dequantize_block(quantized_block, quantized_scale, scale_of_scales):
    """
    Rekonstruiert die ursprünglichen Gewichte
    """
    # Erst den Scale dequantisieren
    reconstructed_scale = quantized_scale * scale_of_scales
    
    # Dann die Gewichte dequantisieren
    reconstructed_weights = quantized_block.astype(np.float32) * reconstructed_scale
    
    return reconstructed_weights, reconstructed_scale

# Dequantisiere alle Blöcke
reconstructed_weights = []
reconstruction_errors = []

for i, (q_block, q_scale) in enumerate(zip(quantized_blocks, quantized_scales)):
    recon_block, recon_scale = dequantize_block(q_block, q_scale, scale_of_scales)
    reconstructed_weights.extend(recon_block)
    
    # Berechne Fehler
    original_block = blocks[i]
    error = np.abs(original_block - recon_block).mean()
    reconstruction_errors.append(error)
    
    print(f"\nBlock {i+1}:")
    print(f"  Original Scale: {scales_fp32[i]:.6f}")
    print(f"  Rekonstruierter Scale: {recon_scale:.6f}")
    print(f"  Scale Fehler: {abs(scales_fp32[i] - recon_scale):.6f}")
    print(f"  Mittlerer Gewichts-Fehler: {error:.6f}")

reconstructed_weights = np.array(reconstructed_weights)

# Berechne Gesamtfehler
total_error = np.abs(original_weights - reconstructed_weights).mean()
relative_error = total_error / np.abs(original_weights).mean() * 100

print(f"\n📊 Gesamtfehler der Rekonstruktion:")
print(f"  Mittlerer absoluter Fehler: {total_error:.6f}")
print(f"  Relativer Fehler: {relative_error:.2f}%")

# ============================================================================
# Visualisierungen
# ============================================================================
print("\n📈 Erstelle Visualisierungen...")

# Erstelle eine große Figure mit mehreren Subplots
fig = plt.figure(figsize=(16, 12))

# 1. Vergleich Original vs Rekonstruiert
ax1 = plt.subplot(3, 2, 1)
sample_indices = np.arange(100)
ax1.plot(sample_indices, original_weights[:100], 'b-', label='Original', linewidth=2, alpha=0.7)
ax1.plot(sample_indices, reconstructed_weights[:100], 'r--', label='Rekonstruiert', linewidth=2, alpha=0.7)
ax1.set_xlabel('Gewichts-Index')
ax1.set_ylabel('Wert')
ax1.set_title('Original vs. Rekonstruierte Gewichte (erste 100)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Fehlerverteilung
ax2 = plt.subplot(3, 2, 2)
errors = np.abs(original_weights - reconstructed_weights)
ax2.hist(errors, bins=50, color='orange', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Absoluter Fehler')
ax2.set_ylabel('Häufigkeit')
ax2.set_title(f'Fehlerverteilung (Mittel: {total_error:.6f})')
ax2.grid(True, alpha=0.3)

# 3. Blockvisualisierung
ax3 = plt.subplot(3, 2, 3)
for i in range(num_blocks):
    start_idx = i * BLOCK_SIZE
    color = plt.cm.viridis(i / num_blocks)
    ax3.axvspan(start_idx, start_idx + BLOCK_SIZE, alpha=0.3, color=color, label=f'Block {i+1}')
ax3.plot(original_weights, 'k-', linewidth=1, alpha=0.5)
ax3.set_xlabel('Gewichts-Index')
ax3.set_ylabel('Wert')
ax3.set_title(f'Blockstruktur (Blockgröße: {BLOCK_SIZE})')
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Scales Vergleich
ax4 = plt.subplot(3, 2, 4)
block_nums = np.arange(1, num_blocks + 1)
reconstructed_scales = quantized_scales * scale_of_scales
ax4.plot(block_nums, scales_fp32, 'bo-', label='Original (FP32)', markersize=8, linewidth=2)
ax4.plot(block_nums, reconstructed_scales, 'rs--', label='Rekonstruiert (8-Bit)', markersize=8, linewidth=2)
ax4.set_xlabel('Block Nummer')
ax4.set_ylabel('Scale Wert')
ax4.set_title('Skalierungsfaktoren: Original vs. Double-Quantisiert')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Speichervergleich
ax5 = plt.subplot(3, 2, 5)
categories = ['Original\n(FP32)', 'Nach 4-Bit\nQuantisierung', 'Nach Double\nQuantization']
sizes = [original_weights.nbytes, total_size_after_q1, total_size_after_dq]
colors = ['#ff6b6b', '#ffa500', '#4ecdc4']
bars = ax5.bar(categories, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('Bytes')
ax5.set_title('Speichernutzung im Vergleich')
ax5.grid(True, alpha=0.3, axis='y')

# Füge Werte auf den Balken hinzu
for bar, size in zip(bars, sizes):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{size:.1f} B\n({original_weights.nbytes/size:.1f}x)',
            ha='center', va='bottom', fontweight='bold')

# 6. Quantisierte Werte Heatmap
ax6 = plt.subplot(3, 2, 6)
# Erstelle eine Matrix aus den quantisierten Blöcken
quant_matrix = np.array([q_block for q_block in quantized_blocks])
im = ax6.imshow(quant_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
ax6.set_xlabel('Position im Block')
ax6.set_ylabel('Block Nummer')
ax6.set_title('4-Bit Quantisierte Werte (Heatmap)')
plt.colorbar(im, ax=ax6, label='Quantisierter Wert')

plt.tight_layout()
plt.savefig('qlora_quantization_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Visualisierung gespeichert!")

# ============================================================================
# Zusammenfassung
# ============================================================================
print("\n" + "=" * 80)
print("📋 ZUSAMMENFASSUNG")
print("=" * 80)

print(f"""
Quantisierung erfolgreich durchgeführt!

🔢 Parameter:
  - Anzahl Gewichte: {len(original_weights)}
  - Blockgröße: {BLOCK_SIZE}
  - Anzahl Blöcke: {num_blocks}

💾 Speicher:
  - Original (FP32): {original_weights.nbytes} Bytes
  - Nach 4-Bit Quantisierung: {total_size_after_q1:.1f} Bytes ({original_weights.nbytes/total_size_after_q1:.2f}x Kompression)
  - Nach Double Quantization: {total_size_after_dq:.1f} Bytes ({original_weights.nbytes/total_size_after_dq:.2f}x Kompression)
  - Einsparung durch DQ: {total_size_after_q1 - total_size_after_dq:.1f} Bytes

🎯 Genauigkeit:
  - Mittlerer absoluter Fehler: {total_error:.6f}
  - Relativer Fehler: {relative_error:.2f}%

✨ Bei einem 65B Parameter Modell würde Double Quantization
   etwa 0.37 GB zusätzlichen Speicher einsparen!
""")

print("=" * 80)
print("Demo abgeschlossen! Visualisierung wurde erstellt.")
print("=" * 80)

plt.show()