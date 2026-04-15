#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

# Modify the doit.py logic to run only 50 frames for testing
exec(open('doit.py').read().replace(
    'total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))',
    'total_frames = min(50, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))'
))