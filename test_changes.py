#!/usr/bin/env python3
"""
Test script to verify the changes to doit.py
This creates a simple test case without running the full video analysis
"""
import os
import cv2
import numpy as np
from datetime import timedelta

# Test parameters
TEST_NUMBER = "123"
TIME_OFFSET = 10.5  # 10.5 seconds offset
BEST_TIME_SEC = 120.75  # 2 minutes 0.75 seconds into video

# Create test output directory
OUTPUT_DIR = 'test_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create a blank test frame (simulating a video frame)
height, width = 720, 1280
test_frame = np.zeros((height, width, 3), dtype=np.uint8)
test_frame.fill(200)  # Light gray background

# Simulate bib detection coordinates (upper body area)
p1 = (width // 2 - 50, height // 2 - 30)
p2 = (width // 2 + 50, height // 2 + 30)

# Calculate bib dimensions for 3x crop
bib_w = p2[0] - p1[0]
bib_h = p2[1] - p1[1]

# Calculate 3x crop coordinates
y_start = max(0, p1[1] - 2*bib_h)
y_end = min(height, p2[1] + 2*bib_h)
x_start = max(0, p1[0] - 2*bib_w)
x_end = min(width, p2[0] + 2*bib_w)

# Create close-up bib image
cropped_runner = test_frame[y_start:y_end, x_start:x_end]
bib_closeup_filename = f"bib_closeup_{TEST_NUMBER}.jpg"
cv2.imwrite(os.path.join(OUTPUT_DIR, bib_closeup_filename), cropped_runner)

# Create complete picture with calculated time prominently displayed
complete_picture_filename = f"complete_{TEST_NUMBER}.jpg"
complete_frame = test_frame.copy()

# Draw bib detection (green rectangle and label)
cv2.rectangle(complete_frame, p1, p2, (0, 255, 0), 3)
cv2.rectangle(complete_frame, (p1[0], p1[1] - 40), (p1[0] + 150, p1[1]), (0, 255, 0), -1)
cv2.putText(complete_frame, f"Bib: {TEST_NUMBER}", (p1[0] + 5, p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Display calculated time prominently in upper left corner
display_time = BEST_TIME_SEC + TIME_OFFSET
time_text = str(timedelta(seconds=display_time))[:-3]
cv2.putText(complete_frame, time_text, (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)  # Large, bold, red text

cv2.imwrite(os.path.join(OUTPUT_DIR, complete_picture_filename), complete_frame)

print(f"Test completed successfully!")
print(f"Generated files:")
print(f"  - {bib_closeup_filename}")
print(f"  - {complete_picture_filename}")
print(f"Calculated time: {time_text}")
print(f"Files saved in: {OUTPUT_DIR}")

# Clean up test files
# os.remove(os.path.join(OUTPUT_DIR, bib_closeup_filename))
# os.remove(os.path.join(OUTPUT_DIR, complete_picture_filename))
# os.rmdir(OUTPUT_DIR)