import cv2
import numpy as np
from ultralytics import YOLO
import os

# 1. INIZIALIZZAZIONE E CONTROLLI
model = YOLO("yolov8n-pose.pt")
video_path = "videoTest.mp4"

# Verifica se il file esiste prima di iniziare
if not os.path.exists(video_path):
    print(f"ERRORE: Il file {video_path} non è nella cartella del progetto!")
    exit()

cap = cv2.VideoCapture(video_path)

# Lettura parametri video per il salvataggio
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: fps = 30  # Fallback se non rileva i FPS

# Setup del salvataggio video
output_path = "output_sorveglianza.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Variabili di stato per la logica anti-furto
has_reached = False
memory_timer = 0
alert_active = 0


def apply_blur(frame, box):
    """Sfoca solo la parte superiore del rettangolo (il viso)."""
    x1, y1, x2, y2 = map(int, box)
    face_h = int((y2 - y1) * 0.25)  # Prende il 25% superiore
    face_zone = frame[max(0, y1):max(0, y1 + face_h), max(0, x1):max(0, x2)]

    if face_zone.size > 0:
        blurred = cv2.GaussianBlur(face_zone, (51, 51), 30)
        frame[max(0, y1):max(0, y1 + face_h), max(0, x1):max(0, x2)] = blurred
    return frame


print("Elaborazione in corso... Premi 'q' per interrompere.")

# 2. LOOP PRINCIPALE
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fine del video o errore di lettura.")
        break

    # Eseguiamo la Pose Estimation (confidenza 0.3 per non perdere i polsi)
    results = model(frame, conf=0.3, verbose=False)

    for r in results:
        if not r.keypoints or len(r.keypoints.xy[0]) < 13:
            continue

        kp = r.keypoints.xy[0].cpu().numpy()

        # Punti chiave: 5/6=Spalle, 10=Polso DX, 12=Anca DX
        # Calcolo distanza spalle per rendere le soglie adattive
        dist_spalle = np.linalg.norm(kp[5] - kp[6])
        if dist_spalle < 10: dist_spalle = 50

        # Distanza Polso-Anca
        dist_polso_anca = np.linalg.norm(kp[10] - kp[12])

        # FASE 1: Il braccio si allontana (prende l'oggetto)
        if dist_polso_anca > dist_spalle * 1.3:
            has_reached = True
            memory_timer = 40  # Ricorda il movimento per 40 frame

        # FASE 2: Occultamento (Mano torna all'anca dopo essersi allontanata)
        if has_reached and dist_polso_anca < dist_spalle * 0.4:
            alert_active = 90  # Rimuove il blur per 3 secondi
            has_reached = False

        if memory_timer > 0:
            memory_timer -= 1
        else:
            has_reached = False

        # 3. GESTIONE VISIVA (Blur vs Alert)
        if r.boxes:
            box = r.boxes[0].xyxy[0].cpu().numpy()
            if alert_active > 0:
                alert_active -= 1
                # Rettangolo ROSSO (Niente Blur)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
                cv2.putText(frame, "!!! FURTO - ID SBLOCCATO !!!", (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Rettangolo VERDE + BLUR sul viso
                frame = apply_blur(frame, box)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

    # Scrittura del frame nel file di output
    out.write(frame)

    # Mostra a video
    cv2.imshow("Sistema Sorveglianza Privacy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. CHIUSURA PULITA
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Fine. Video salvato correttamente in: {os.path.abspath(output_path)}")