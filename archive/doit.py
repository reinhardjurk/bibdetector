import cv2
import os
import numpy as np
from collections import defaultdict, Counter
from ultralytics import YOLO
import easyocr
from tqdm import tqdm
import torch

# ==========================================================
# CONFIGURATION
# ==========================================================

VIDEO_PATH = "test.short.m4v"
OUTPUT_DIR = "output"
CONTINUOUS_OUTPUT_DIR = "continuous_output"  # Directory for continuous frame output

YOLO_MODEL = "yolov8m.pt"      # stronger model for distant runners
FRAME_SKIP = 1                 # increase to 2–3 for more speed
MIN_OCR_VOTES = 2              # minimum consistent sightings (lowered for testing)
MIN_OCR_CONF = 0.35            # filter weak OCR
FINISH_LINE_RATIO = 0.85       # 85% of image height

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONTINUOUS_OUTPUT_DIR, exist_ok=True)

# ==========================================================
# DEVICE SETUP (Apple Silicon Optimized)
# ==========================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading YOLO...")
model = YOLO(YOLO_MODEL)

print("Loading OCR...")
reader = easyocr.Reader(['en'], gpu=(device != "cpu"))

# ==========================================================
# DATA STRUCTURES
# ==========================================================

track_votes = defaultdict(list)  # Stores bib numbers detected
track_detection_times = defaultdict(list)  # Stores detection times for each track
track_detection_frames = defaultdict(list)  # Stores frames for each detection
track_finish_time = {}
track_crossed = set()

# ==========================================================
# VIDEO SETUP
# ==========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Could not open video file.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

finish_line_y = int(height * FINISH_LINE_RATIO)

print(f"Video resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")

frame_idx = 0

# ==========================================================
# OCR PREPROCESSING FUNCTION
# ==========================================================

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh

# ==========================================================
# MAIN LOOP
# ==========================================================

print("Processing video...")

with tqdm(total=total_frames) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        # Save continuous frame output
        continuous_frame_path = os.path.join(CONTINUOUS_OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(continuous_frame_path, frame)

        # -------- Detect + Track runners --------
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],   # class 0 = person
            device=device,
            verbose=False
        )

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)

                # Crop runner region
                runner_crop = frame[y1:y2, x1:x2]

                if runner_crop.size == 0:
                    continue

                # OCR preprocessing
                processed = preprocess_for_ocr(runner_crop)

                # Run OCR
                ocr_results = reader.readtext(processed)

                bib_detected = False
                detected_bibs = []  # Store all detected bibs in this frame
                
                for (_, text, conf) in ocr_results:

                    if conf < MIN_OCR_CONF:
                        continue

                    digits = ''.join(c for c in text if c.isdigit())

                    if 2 <= len(digits) <= 5:
                        track_votes[track_id].append(digits)
                        bib_detected = True
                        detected_bibs.append(digits)

                        # Store detection time and frame (only keep last few for memory efficiency)
                        detection_time = frame_idx / fps
                        track_detection_times[track_id].append(detection_time)
                        # Only store frames for recent detections to save memory
                        if len(track_detection_frames[track_id]) < 5:  # Keep max 5 frames per track
                            track_detection_frames[track_id].append(frame.copy())

                # Mark bib detection on the frame
                if bib_detected:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "BIB DETECTED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save marked frame
                    marked_frame_path = os.path.join(CONTINUOUS_OUTPUT_DIR, f"frame_{frame_idx:06d}_bib_detected.jpg")
                    cv2.imwrite(marked_frame_path, frame)
                    
                    # Create preview images for each detected bib
                    detection_time = frame_idx / fps
                    for bib_digits in detected_bibs:
                        # Create preview image with bib number and time overlay
                        preview_frame = frame.copy()
                        
                        # Draw semi-transparent overlay background
                        overlay = preview_frame.copy()
                        cv2.rectangle(overlay, (x1, y1 - 40), (x2, y1), (0, 0, 0), cv2.FILLED)
                        alpha = 0.6  # Transparency factor
                        cv2.addWeighted(overlay, alpha, preview_frame, 1 - alpha, 0, preview_frame)
                        
                        # Add bib number and time text
                        bib_text = f"Bib: {bib_digits}"
                        time_text = f"Time: {detection_time:.2f}s"
                        
                        cv2.putText(preview_frame, bib_text, (x1 + 5, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(preview_frame, time_text, (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Save preview image
                        preview_path = os.path.join(CONTINUOUS_OUTPUT_DIR, f"preview_bib_{bib_digits}_time_{detection_time:.2f}s.jpg")
                        cv2.imwrite(preview_path, preview_frame)

                # -------- Finish Line Crossing --------
                center_y = (y1 + y2) // 2

                if (
                    center_y > finish_line_y
                    and track_id not in track_crossed
                ):
                    track_crossed.add(track_id)
                    track_finish_time[track_id] = frame_idx / fps

        frame_idx += 1
        pbar.update(1)
        
        # Clean up OpenCV windows periodically to prevent memory buildup
        if frame_idx % 100 == 0:
            cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

# ==========================================================
# FINALIZE RESULTS (Majority Voting)
# ==========================================================

final_bibs = {}
bib_detection_info = {}  # Store all detection times and middle frame for each bib

for track_id, votes in track_votes.items():
    if len(votes) >= MIN_OCR_VOTES:
        most_common = Counter(votes).most_common(1)[0][0]
        final_bibs[track_id] = most_common
        
        # Get all detection times for this track
        detection_times = track_detection_times[track_id]
        
        # Find the middle detection time and corresponding frame
        middle_index = min(len(detection_times) // 2, len(track_detection_frames[track_id]) - 1)
        middle_time = detection_times[middle_index]
        middle_frame = track_detection_frames[track_id][middle_index]
        
        # Store bib info with all detection times and middle frame
        bib_detection_info[track_id] = {
            'bib': most_common,
            'all_times': detection_times,
            'middle_time': middle_time,
            'middle_frame': middle_frame
        }
        
        # Display preview for the middle detection only
        preview_frame = middle_frame.copy()
        
        # Find the bounding box for this track (we'll use a placeholder since we don't have the exact box here)
        # For simplicity, we'll just add text overlay to the middle frame
        bib_text = f"Bib: {most_common}"
        time_text = f"Time: {middle_time:.2f}s"
        
        # Draw semi-transparent overlay background at top
        overlay = preview_frame.copy()
        cv2.rectangle(overlay, (0, 0), (preview_frame.shape[1], 40), (0, 0, 0), cv2.FILLED)
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, preview_frame, 1 - alpha, 0, preview_frame)
        
        # Add bib number and time text
        cv2.putText(preview_frame, bib_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(preview_frame, time_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display preview for middle detection
        cv2.imshow(f"Bib {most_common} - Middle Detection {middle_time:.2f}s", preview_frame)
        cv2.waitKey(1000)  # Display for 1 second

print(f"Final recognized bibs: {len(final_bibs)}")

# ==========================================================
# SAVE IMAGES + GENERATE HTML
# ==========================================================

html_lines = []
html_lines.append("<html><body>")
html_lines.append("<h1>Marathon Finish Results</h1>")
html_lines.append("<ul>")

saved_bibs = set()

for track_id, bib_info in bib_detection_info.items():
    bib = bib_info['bib']
    
    if bib in saved_bibs:
        continue   # ensure only one image per bib

    saved_bibs.add(bib)

    # Save the middle detection frame
    img_filename = f"bib_{bib}.jpg"
    img_path = os.path.join(OUTPUT_DIR, img_filename)
    cv2.imwrite(img_path, bib_info['middle_frame'])

    # Format all detection times
    all_times_str = ", ".join([f"{t:.2f}s" for t in bib_info['all_times']])
    middle_time_str = f"{bib_info['middle_time']:.2f}s"

    html_lines.append(f'<li>Bib {bib} - Detected at: {all_times_str}</li>')
    html_lines.append(f'<li style="margin-left: 20px;">Middle detection: {middle_time_str} - '
                     f'<a href="{img_filename}">View Image</a></li>')

html_lines.append("</ul></body></html>")

html_path = os.path.join(OUTPUT_DIR, "results.html")

with open(html_path, "w") as f:
    f.write("\n".join(html_lines))

print("-------------------------------------------------")
print(f"Results saved to: {html_path}")
print("Done.")
