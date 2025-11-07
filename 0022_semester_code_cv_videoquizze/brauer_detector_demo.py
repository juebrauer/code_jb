from ultralytics import YOLO
import cv2
import time
import math

model = YOLO("/home/juebrauer/link_to_vcd/10_datasets/61_brauer_dataset_yolo/runs/detect/train3/weights/best.pt")

# 0 = Standard-Webcam. Je nach System kann es 1, 2, ... sein.
cap = cv2.VideoCapture(4)

# Optional: Auflösung setzen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Kamera konnte nicht geöffnet werden.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kein Frame von der Kamera erhalten.")
            break

        # Inferenz (du kannst imgsz / device / half etc. anpassen)
        results = model(frame, verbose=False)

        # Annotiertes Bild holen und anzeigen
        annot = results[0].plot()
        
        cv2.imshow("Brauerdetector Demo", annot)

        # Mit 'q' beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
