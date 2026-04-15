import cv2
import easyocr
import os
from datetime import timedelta
from ultralytics import YOLO
import math

# --- Configuration ---
VIDEO_PATH = 'GP019726.mp4'  
OUTPUT_DIR = 'runner_data'   
HTML_FILE = 'race_results.html'
FRAME_SKIP = 2               
PATIENCE_FRAMES = 30         
CONFIDENCE_THRESHOLD = 0.5   
MIN_CONSECUTIVE_DETECTIONS = 2 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ABS_VIDEO_PATH = os.path.abspath(VIDEO_PATH)

# --- HTML Initialization ---
with open(HTML_FILE, 'w', encoding='utf-8') as f:
    f.write("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Race - Live Tracking</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; padding: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }
            th, td { padding: 12px; border: 1px solid #ddd; text-align: left; vertical-align: middle; }
            th { background-color: #007BFF; color: white; }
            img { max-width: 300px; border-radius: 5px; }
            .vlc-btn { background-color: #ff8800; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; font-weight: bold; }
            .vlc-btn:hover { background-color: #cc6c00; }
        </style>
    </head>
    <body>
        <h1>Detected Runners</h1>
        <table>
            <tr>
                <th>Bib Number</th>
                <th>Time (hh:mm:ss)</th>
                <th>Snapshot</th>
                <th>Playback</th>
            </tr>
    """)

def append_to_html(number, timestamp_str, img_filename, m3u_filename):
    with open(HTML_FILE, 'a', encoding='utf-8') as f:
        f.write(f"""
            <tr>
                <td><strong>{number}</strong></td>
                <td>{timestamp_str}</td>
                <td><img src="{OUTPUT_DIR}/{img_filename}" alt="Bib {number}"></td>
                <td><a href="{OUTPUT_DIR}/{m3u_filename}" class="vlc-btn">▶ Play in VLC</a></td>
            </tr>
        """)

def create_m3u_playlist(m3u_path, start_time_sec):
    with open(m3u_path, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")
        f.write(f"#EXTVLCOPT:start-time={max(0, start_time_sec)} \n")
        f.write(f"file://{ABS_VIDEO_PATH}\n")

# --- Model Initialization ---
print("Loading YOLOv8 tracking model...")
yolo_model = YOLO('yolov8n.pt') 

print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=True) 

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# --- ROI Selection (Region of Interest) ---
ret, first_frame = cap.read()
if not ret:
    print("Fehler: Video konnte nicht gelesen werden.")
    exit()

print("Bitte wähle den Zielbereich (ROI) im Fenster aus und drücke ENTER oder LEERTASTE.")
print("Drücke 'c' um die Auswahl abzubrechen und das gesamte Bild zu nutzen.")

# Fenster für die ROI-Auswahl öffnen
roi = cv2.selectROI("Select ROI - Press Enter to confirm", first_frame, fromCenter=False, showCrosshair=True)
r_x, r_y, r_w, r_h = roi

cv2.destroyWindow("Select ROI - Press Enter to confirm")
cv2.waitKey(1) # Wichtig auf dem Mac, damit sich das Fenster wirklich schließt

if r_w == 0 or r_h == 0:
    print("Keine ROI ausgewählt. Verarbeite das gesamte Bild.")
    r_x, r_y, r_w, r_h = 0, 0, first_frame.shape[1], first_frame.shape[0]
else:
    print(f"ROI ausgewählt: X={r_x}, Y={r_y}, Width={r_w}, Height={r_h}")

# Setze das Video wieder auf Frame 0 zurück
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

active_tracks = {} 
processed_track_ids = set() 
globally_completed_bibs = set() 
frame_idx = 0

print("Starting video analysis...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_idx += 1
    current_time_sec = frame_idx / fps
    current_timestamp_str = str(timedelta(seconds=current_time_sec))[:-3]

    if frame_idx % FRAME_SKIP != 0:
        continue

    # Schneide das Bild auf die ROI zu
    roi_frame = frame[r_y:r_y+r_h, r_x:r_x+r_w]

    # 1. Run YOLO Tracking (NUR auf dem zugeschnittenen ROI-Bild)
    yolo_results = yolo_model.track(
        roi_frame, persist=True, classes=[0], tracker="bytetrack.yaml", device='mps', verbose=False
    )

    if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        track_ids = yolo_results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            if track_id in processed_track_ids:
                continue

            px1, py1, px2, py2 = map(int, box)
            h, w = roi_frame.shape[:2]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            if (px2 - px1) < 60 or (py2 - py1) < 120:
                continue

            person_crop = roi_frame[py1:py2, px1:px2]
            ocr_results = reader.readtext(person_crop)

            found_bibs_in_crop = []
            for (ocr_bbox, text, prob) in ocr_results:
                clean_number = ''.join(filter(str.isdigit, text))
                
                if clean_number in globally_completed_bibs:
                    continue

                if clean_number and prob > CONFIDENCE_THRESHOLD:
                    # Koordinaten-Magie: Wir rechnen die Koordinaten von der OCR 
                    # zurück auf den Person-Crop, dann auf die ROI und dann auf das Gesamtbild.
                    abs_p1 = (int(ocr_bbox[0][0]) + px1 + r_x, int(ocr_bbox[0][1]) + py1 + r_y)
                    abs_p2 = (int(ocr_bbox[2][0]) + px1 + r_x, int(ocr_bbox[2][1]) + py1 + r_y)
                    absolute_bbox = [abs_p1, abs_p2]
                    
                    found_bibs_in_crop.append((absolute_bbox, clean_number))

            if track_id not in active_tracks:
                active_tracks[track_id] = {'sightings': [], 'last_seen_frame': frame_idx}
            
            active_tracks[track_id]['last_seen_frame'] = frame_idx
            
            for bbox, number in found_bibs_in_crop:
                # Wir speichern das *gesamte* Frame für den Export, damit man im HTML das ganze Bild sieht
                active_tracks[track_id]['sightings'].append(
                    (frame.copy(), current_timestamp_str, bbox, number, current_time_sec)
                )

    # --- 2. Pruning & Consecutive Logic ---
    tracks_to_process = []
    for t_id, data in active_tracks.items():
        if (frame_idx - data['last_seen_frame']) > PATIENCE_FRAMES:
            tracks_to_process.append(t_id)

    for t_id in tracks_to_process:
        data = active_tracks[t_id]
        sightings = data['sightings']
        
        if len(sightings) >= MIN_CONSECUTIVE_DETECTIONS:
            best_num = None
            max_consecutive = 0
            current_num = None
            current_count = 0
            
            for s in sightings:
                num = s[3] 
                if num == current_num:
                    current_count += 1
                else:
                    if current_count > max_consecutive:
                        max_consecutive = current_count
                        best_num = current_num
                    current_num = num
                    current_count = 1
            
            if current_count > max_consecutive:
                max_consecutive = current_count
                best_num = current_num

            if max_consecutive >= MIN_CONSECUTIVE_DETECTIONS and best_num not in globally_completed_bibs:
                
                valid_sightings = [s for s in sightings if s[3] == best_num]
                mid_index = math.floor(len(valid_sightings) * 0.75)
                best_frame, best_timestamp, best_bbox, final_number, best_time_sec = valid_sightings[mid_index]
                
                p1, p2 = best_bbox
                h_img, w_img = best_frame.shape[:2]

                # --- TRIPLE SIZE CROP LOGIC ---
                # 1. Calculate the current bib dimensions
                bib_w = p2[0] - p1[0]
                bib_h = p2[1] - p1[1]

                # 2. Expand the box to 3x the size (adding 1x the size to each side)
                # This ensures the bib remains centered in a larger context
                y_start = max(0, p1[1] - 2*bib_h)
                y_end = min(h_img, p2[1] + 2*bib_h)
                x_start = max(0, p1[0] - 2*bib_w)
                x_end = min(w_img, p2[0] + 2*bib_w)

                # 3. Draw the recognition UI on the frame *before* cropping
                cv2.rectangle(best_frame, p1, p2, (0, 255, 0), 3)
                cv2.rectangle(best_frame, (p1[0], p1[1] - 40), (p1[0] + 150, p1[1]), (0, 255, 0), -1)
                label = f"Bib: {final_number}"
                cv2.putText(best_frame, label, (p1[0] + 5, p1[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # 4. Perform the crop
                cropped_runner = best_frame[y_start:y_end, x_start:x_end]
                # ------------------------------

                img_filename = f"runner_{final_number}.jpg"
                m3u_filename = f"play_{final_number}.m3u"
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, img_filename), cropped_runner)
                
                create_m3u_playlist(os.path.join(OUTPUT_DIR, m3u_filename), best_time_sec - 2.0) 
                append_to_html(final_number, best_timestamp, img_filename, m3u_filename)
                
                print(f"-> Verified Bib {final_number} (3x Context Crop). Saved!")
                globally_completed_bibs.add(final_number)
        
        processed_track_ids.add(t_id)
        del active_tracks[t_id]

with open(HTML_FILE, 'a', encoding='utf-8') as f:
    f.write("</table></body></html>")

cap.release()
print(f"Total unique runners identified: {len(globally_completed_bibs)}")
print("Analysis complete. Open 'race_results.html' in your browser.")
