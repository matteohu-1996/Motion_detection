import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import deque

# 1. INIZIALIZZAZIONE E CONTROLLI
model = YOLO("yolov8n-pose.pt")
video_path = "videoTest.mp4"

if not os.path.exists(video_path):
    print(f"ERRORE: Il file {video_path} non è nella cartella del progetto!")
    exit()

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: fps = 30

# --- CONFIGURAZIONE SALVATAGGIO DINAMICO ---
output_path = "EVIDENZA_FURTO.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None  # Non inizializziamo subito il file su disco

# Buffer circolare per i 5 secondi precedenti (fps * 5)
pre_record_buffer = deque(maxlen=fps * 5)
is_recording = False
post_record_counter = 0

# --- GESTIONE MULTI-PERSONA ---
dict_has_reached = {}
dict_memory_timer = {}
dict_alert_active = {}


def apply_blur(frame, box):
    """Sfoca solo la parte superiore del rettangolo (il viso)."""
    x1, y1, x2, y2 = map(int, box)
    face_h = int((y2 - y1) * 0.25)
    face_zone = frame[max(0, y1):max(0, y1 + face_h), max(0, x1):max(0, x2)]
    if face_zone.size > 0:
        blurred = cv2.GaussianBlur(face_zone, (51, 51), 30)
        frame[max(0, y1):max(0, y1 + face_h), max(0, x1):max(0, x2)] = blurred
    return frame


print("Monitoraggio attivo... Salvataggio solo in caso di furto rilevato.")

# 2. LOOP PRINCIPALE
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Copia pulita per il buffer (con eventuale blur ma senza scritte di allerta)
    frame_for_buffer = frame.copy()

    results = model.track(frame, conf=0.25, persist=True, tracker="bytetrack.yaml", verbose=False)
    theft_trigger_now = False  # Variabile per attivare la scrittura immediata

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        keypoints_data = results[0].keypoints.xy.cpu().numpy()
        conf_data = results[0].keypoints.conf.cpu().numpy()

        for i, track_id in enumerate(ids):
            if track_id not in dict_has_reached:
                dict_has_reached[track_id] = False
                dict_memory_timer[track_id] = 0
                dict_alert_active[track_id] = 0

            kp = keypoints_data[i]
            conf = conf_data[i]
            box = boxes[i]

            if len(kp) > 12 and conf[5] > 0.3 and conf[6] > 0.3:
                dist_spalle = np.linalg.norm(kp[5] - kp[6])
                if dist_spalle < 10: dist_spalle = 50

                ref_point = kp[12] if conf[12] > 0.3 else kp[6]
                dist_azione = np.linalg.norm(kp[10] - ref_point)

                # LOGICA FURTO (Tolleranze richieste)
                if dist_azione > dist_spalle * 1.2:
                    dict_has_reached[track_id] = True
                    dict_memory_timer[track_id] = 40

                if dict_has_reached[track_id] and dist_azione < dist_spalle * 0.3:
                    dict_alert_active[track_id] = 90
                    dict_has_reached[track_id] = False
                    theft_trigger_now = True  # TRIGGER DI SICUREZZA

                if dict_memory_timer[track_id] > 0:
                    dict_memory_timer[track_id] -= 1
                else:
                    dict_has_reached[track_id] = False

            # GESTIONE VISIVA
            x1, y1, x2, y2 = map(int, box)
            label = f"ID: {track_id}"

            if dict_alert_active[track_id] > 0:
                dict_alert_active[track_id] -= 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"{label} - !!! FURTO !!!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame = apply_blur(frame, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- LOGICA DI SALVATAGGIO SELETTIVO ---
    if theft_trigger_now and not is_recording:
        print("!!! FURTO RILEVATO: Inizio salvataggio video (inclusi 5s precedenti) !!!")
        is_recording = True
        post_record_counter = fps * 5  # Registra altri 5 secondi dopo il furto
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Svuota il buffer dei 5 secondi passati nel file video
        while pre_record_buffer:
            out.write(pre_record_buffer.popleft())

    if is_recording:
        out.write(frame)
        post_record_counter -= 1
        if post_record_counter <= 0:
            print("Salvataggio completato.")
            is_recording = False
            out.release()
            out = None
    else:
        # Se non stiamo registrando, aggiungiamo il frame al buffer circolare
        pre_record_buffer.append(frame_for_buffer)

    cv2.imshow("Sistema Sorveglianza Multi-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
if out: out.release()
cv2.destroyAllWindows()