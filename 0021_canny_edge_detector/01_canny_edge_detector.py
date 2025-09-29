import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a simple synthetic image (rectangle + circle)
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (100, 100), 40, 127, -1)

# Step 1: Smooth image using Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(img, (5,5), 1.4)

# Step 2: Compute gradients using Sobel operators
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)
magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

# Step 3+4: Apply Canny edge detection (includes Non-Maximum Suppression + Hysteresis)
edges = cv2.Canny(img, 50, 150)


fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(blurred, cmap='gray')
axes[0, 1].set_title("After Gaussian Blur")
axes[0, 1].axis("off")

axes[0, 2].imshow(magnitude, cmap='gray')
axes[0, 2].set_title("Gradient Magnitude")
axes[0, 2].axis("off")

axes[1, 0].imshow(grad_x, cmap='gray')
axes[1, 0].set_title("Sobel X (dI/dx)")
axes[1, 0].axis("off")

axes[1, 1].imshow(grad_y, cmap='gray')
axes[1, 1].set_title("Sobel Y (dI/dy)")
axes[1, 1].axis("off")

axes[1, 2].imshow(edges, cmap='gray')
axes[1, 2].set_title("Canny Edges")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()