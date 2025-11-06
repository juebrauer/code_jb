from ultralytics import YOLO
import cv2
import time
import math

model = YOLO("yolo11n-pose.pt")

# 0 = Standard-Webcam. Je nach System kann es 1, 2, ... sein.
cap = cv2.VideoCapture(0)

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
        
        lx,ly = results[0].keypoints.data.cpu().numpy()[0][9][:2]
        lx,ly = int(lx), int(ly)
        rx,ry = results[0].keypoints.data.cpu().numpy()[0][10][:2]
        rx,ry = int(rx), int(ry)
        
        
        cv2.circle(annot, (lx,ly), radius=5, color=(255,255,255), thickness=5)
        cv2.circle(annot, (rx,ry), radius=5, color=(0,0,0), thickness=5)

        dx = lx - rx
        dy = ly - ry
        
        dist = int(math.sqrt( dx**2 + dy**2 ))

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        
        comment = ""
        if (dist < 100):
            comment = "Bitte halten Sie die Haende auseinander!"

        cv2.putText(annot, f"{dist} {comment}", (20,30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Pose - Live", annot)

        # Mit 'q' beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
