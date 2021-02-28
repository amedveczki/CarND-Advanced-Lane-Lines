# -*- coding: utf-8 -*-

# Plan:
# 1. Camera calibration
#  - grayscale conversion
#  - find chessboard

# 2. Undistort using camera matrix (9x6 chessboard)
# 3. Perspective transformation (with hardcoded src-dst coordinates)
# 4. Color threshold + Gradient threshold
#  - HSL vs HSV vs..., check which would be best for what

# 5. Detect left/right lane pixels (candidates)
#  a. First run or after too many failures
#   - Create histogram based on the lower part of the image
#   - Starting X will be the highest values in the histogram on left/right side
#   - Capture left/right pixels in the current window (which we will slide upwards)
#   - Use mean X pixel value as next starting point for next window
#  b. Last frame was OK
#   - Use last polynomials for windowing

# 6. Sanity check
#  - If failed, skip to draw (after too many failed frames, reset and try step 5
#  - check for similar curvature (how?)
#  - Checking that they are separated by approximately the right distance horizontally
#  - Checking that they are roughly parallel

# 7. Try to fit polynomial on the lanes
#  - if success, in next frame we use this for windowing
#  - Measure curvature (should be ~1km)
 
# 8. Draw lanes and fill with color

from calibration import calibration
import glob

CALIBRATION_IMAGES = "camera_cal/*.jpg"

calib = calibration(True, True)
calib.calibrate(glob.glob(CALIBRATION_IMAGES), 9, 6)
calib.undistort(mpimg.imread(glob.glob(CALIBRATION_IMAGES)[1]))