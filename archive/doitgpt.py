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

YOLO_MODEL = "yolov8m.pt"   # stronger model
FRAME_SKIP = 1
MIN_OCR_VOTES = 3
MIN_OCR_CONF = 0.35
FINISH_LINE_RATIO = 0.85

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# DEVICE SETUP (Apple Silicon Optimized)
# ==========================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading YOLO model...")
model = YOLO(YOLO_MODEL)

print("Loading OCR model...")
reader = easyocr.Reader(['en'], gpu=(device != "cpu"))

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

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")

# ==========================================================
# DATA STRUCTURES
# ==========================================================

track_votes = defaultdict(list)
track_best_frame = {}
track_finish_time = {}
track_crossed = set()

# ==========================================================
# OCR PREPROCESSING
# ==========================================================

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15,
        10
    )

    return thresh

# ==========================================================
# MAIN PROCESSING LOOP
# ==========================================================

frame_idx = 0

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

        # ------------------------------------------------------
        # PERSON DETECTION + TRACKING
        # ------------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],  # person class
            device=device,
            verbose=False
        )

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):

                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)

                box_h = y2 - y1
                box_w = x2 - x1

                # ------------------------------------------------------
                # EXPAND BOX DOWNWARD (CRITICAL FIX)
                # ------------------------------------------------------

                y2_expanded = int(y2 + 0.5 * box_h)
                x1_expanded = int(x1 - 0.1 * box_w)
                x2_expanded = int(x2 + 0.1 * box_w)

                x1_expanded = max(0, x1_expanded)
                y1_expanded = max(0, y1)
                x2_expanded = min(width, x2_expanded)
                y2_expanded = min(height, y2_expanded)

                runner_crop = frame[
                    y1_expanded:y2_expanded,
                    x1_expanded:x2_expanded
                ]

                if runner_crop.size == 0:
                    continue

                # ------------------------------------------------------
                # UPSCALE FOR OCR (IMPORTANT AT 5M DISTANCE)
                # ------------------------------------------------------

                runner_crop = cv2.resize(
                    runner_crop,
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

                processed = preprocess_for_ocr(runner_crop)

                # ------------------------------------------------------
                # OCR
                # ------------------------------------------------------

                ocr_results = reader.readtext(processed)

                for (_, text, conf) in ocr_results:

                    if conf < MIN_OCR_CONF:
                        continue

                    digits = ''.join(c for c in text if c.isdigit())

                    if 2 <= len(digits) <= 5:
                        track_votes[track_id].append(digits)

                        if track_id not in track_best_frame:
                            track_best_frame[track_id] = frame.copy()

                # ------------------------------------------------------
                # FINISH LINE DETECTION
                # ------------------------------------------------------

                center_y = (y1 + y2) // 2

                if (
                    center_y > finish_line_y
                    and track_id not in track_crossed
                ):
                    track_crossed.add(track_id)
                    track_finish_time[track_id] = frame_idx / fps

        frame_idx += 1
        pbar.update(1)

cap.release()

# ==========================================================
# MAJORITY VOTING
# ==========================================================

final_bibs = {}

for track_id, votes in track_votes.items():
    if len(votes) >= MIN_OCR_VOTES:
        final_bibs[track_id] = Counter(votes).most_common(1)[0][0]

print(f"Recognized bibs: {len(final_bibs)}")

# ==========================================================
# SAVE RESULTS
# ==========================================================

html_lines = []
html_lines.append("<html><body>")
html_lines.append("<h1>Marathon Finish Results</h1>")
html_lines.append("<ul>")

saved_bibs = set()

for track_id, bib in final_bibs.items():

    if track_id not in track_finish_time:
        continue

    if bib in saved_bibs:
        continue

    saved_bibs.add(bib)

    img_name = f"bib_{bib}.jpg"
    img_path = os.path.join(OUTPUT_DIR, img_name)

    cv2.imwrite(img_path, track_best_frame[track_id])

    elapsed = track_finish_time[track_id]

    html_lines.append(
        f'<li>Bib {bib} - {elapsed:.2f} sec - '
        f'<a href="{img_name}">Image</a></li>'
    )

html_lines.append("</ul></body></html>")

with open(os.path.join(OUTPUT_DIR, "results.html"), "w") as f:
    f.write("\n".join(html_lines))

print("Done. Results saved to output/results.html")
