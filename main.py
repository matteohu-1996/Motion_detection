import cv2
import numpy as np
from ultralytics import YOLO

# 1. CONFIGURAZIONE
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("videoTest.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = None
target_id = None  # L'ID della persona che fa scattare l'allarme
closing_frames = 0  # Contatore per i 5 secondi di coda
users = {}  # Stato per ogni ID: [ha_allungato, timer, allerta]

print("Sistema in ascolto... La registrazione partirà al primo furto rilevato.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, verbose=False, conf=0.3)
    current_ids = []

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
        kpts = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].keypoints.conf.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        current_ids = ids

        for i, track_id in enumerate(ids):
            if track_id not in users: users[track_id] = [False, 0, 0]

            kp, conf = kpts[i], confs[i]
            # Logica Distanze (Spalle punto 5-6, Polso 10, Anca 12)
            if conf[5] > 0.3 and conf[10] > 0.3:
                dist_spalle = np.linalg.norm(kp[5] - kp[6]) or 50
                ref = kp[12] if conf[12] > 0.3 else kp[6]
                dist_azione = np.linalg.norm(kp[10] - ref)

                # FASE 1: Allungamento
                if dist_azione > dist_spalle * 1.2:
                    users[track_id][0], users[track_id][1] = True, 40

                # FASE 2: Occultamento (Trigger Furto)
                if users[track_id][0] and dist_azione < dist_spalle * 0.3:
                    users[track_id][2] = 100  # Mostra testo a schermo
                    users[track_id][0] = False

                    # Inizia a registrare solo se è il primo furto rilevato
                    if target_id is None:
                        target_id = track_id
                        out = cv2.VideoWriter("FILMATO_FURTO.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        print(f"!!! FURTO RILEVATO (ID {target_id}). REGISTRAZIONE AVVIATA !!!")

                if users[track_id][1] > 0:
                    users[track_id][1] -= 1
                else:
                    users[track_id][0] = False

            # Disegno per tutti gli ID presenti
            is_alert = users[track_id][2] > 0
            color = (0, 0, 255) if is_alert else (0, 255, 0)
            if is_alert: users[track_id][2] -= 1

            bx = boxes[i]
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            cv2.putText(frame, f"ID:{track_id} {'!! FURTO !!' if is_alert else ''}",
                        (bx[0], bx[1] - 10), 0, 0.6, color, 2)

    # 2. LOGICA DI REGISTRAZIONE CONTINUA E USCITA
    if target_id is not None:
        # Scrive sempre il frame corrente
        out.write(frame)

        # Se la persona (ID target) esce dal campo visivo
        if target_id not in current_ids:
            if closing_frames == 0:
                print(f"ID {target_id} uscito. Registro gli ultimi 5 secondi...")

            closing_frames += 1

            # Se abbiamo raggiunto i 5 secondi di coda (fps * 5)
            if closing_frames >= fps * 5:
                print("Registrazione terminata con successo.")
                break  # Interrompe il programma

    cv2.imshow("Sorveglianza Smart", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 3. PULIZIA
cap.release()
if out: out.release()
cv2.destroyAllWindows()