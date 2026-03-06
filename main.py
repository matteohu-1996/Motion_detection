import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# 1. SETUP MODELLO E TRACKER AD ALTA PERSISTENZA
model = YOLO("yolo11s-pose.pt")

cap = cv2.VideoCapture("videoTest.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Buffer pre-furto (5 secondi)
pre_buffer = deque(maxlen=fps * 5)

# Variabili di stato
out = None
target_id = None
recording_active = False
lost_counter = 0

users = {}
id_map = {}
next_human_id = 1

print("--- Sistema di sorveglianza attivato ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_raw = frame.copy()

    # TRACKER RINFORZATO (BoT-SORT con persistenza)
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        conf=0.35,
        tracker="botsort.yaml",
        imgsz=640
    )

    current_ids = []

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
        kpts = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].keypoints.conf.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        current_ids = ids

        for i, track_id in enumerate(ids):
            # Mappatura ID (Persona 1, 2, 3...)
            if track_id not in id_map:
                id_map[track_id] = next_human_id
                next_human_id += 1

            display_id = id_map[track_id]

            if track_id not in users:
                users[track_id] = [False, 0, 0, boxes[i]]

            # Smoothing del box per eliminare il lampeggio
            old_box = users[track_id][3]
            new_box = boxes[i]
            smooth_box = (old_box * 0.85 + new_box * 0.15).astype(int)
            users[track_id][3] = smooth_box

            kp, conf = kpts[i], confs[i]

            # LOGICA FURTO (Allunga -> Ritira)
            if conf[5] > 0.4 and conf[10] > 0.4:
                dist_spalle = np.linalg.norm(kp[5] - kp[6]) or 50
                ref = kp[12] if conf[12] > 0.4 else kp[6]
                dist_azione = np.linalg.norm(kp[10] - ref)

                if dist_azione > dist_spalle * 1.25:
                    users[track_id][0], users[track_id][1] = True, 60

                if users[track_id][0] and dist_azione < dist_spalle * 0.35:
                    users[track_id][2] = 100  # Durata allerta visiva
                    users[track_id][0] = False

                    # Avvio registrazione se non già attiva
                    if not recording_active:
                        target_id = track_id
                        timestamp = int(time.time())
                        filename = f"output_{display_id}_{timestamp}.mp4"
                        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        recording_active = True
                        print(f"FURTO RILEVATO: Persona {display_id} - Registrazione: {filename}")

                        while pre_buffer:
                            out.write(pre_buffer.popleft())

                if users[track_id][1] > 0:
                    users[track_id][1] -= 1
                else:
                    users[track_id][0] = False

            # --- MODIFICA BORDI E COLORI ---
            alert_active = users[track_id][2] > 0

            # Se in allerta: Rosso e spessore 2. Altrimenti: Verde e spessore 1.
            color = (0, 0, 255) if alert_active else (0, 255, 0)
            thickness = 2 if alert_active else 1

            if alert_active: users[track_id][2] -= 1

            b = smooth_box
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, thickness)
            cv2.putText(frame, f"Persona {display_id}", (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # GESTIONE REGISTRAZIONE
    if recording_active:
        out.write(frame)
        if target_id not in current_ids:
            lost_counter += 1
            if lost_counter >= fps * 5:
                print(f"Persona {id_map[target_id]} uscita. Standby...")
                out.release()
                out = None
                recording_active = False
                target_id = None
                lost_counter = 0
        else:
            lost_counter = 0
    else:
        pre_buffer.append(frame_raw)

    cv2.imshow("CCTV", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
if out: out.release()
cv2.destroyAllWindows()