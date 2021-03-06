# -*- coding: utf-8 -*-

# Plan:
# 1. Camera calibration (9x6 chessboard)
#  - grayscale conversion
#  - find chessboard

# 2. Undistort using camera matrix
# 3. Color threshold + Gradient threshold
#  - HSL vs HSV vs..., check which would be best for what
# 4. Perspective transformation (with hardcoded src-dst coordinates)
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
from thresholds import color_sobel_threshold
from perspective import perspective
from lane_detector import lane_detector
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt

import numpy as np
import cv2


class fancy_saver:
    def __init__(self, path = "fancy/"):
        self.path = path
        self.frame = 0
        self.step = 0
    
    def next_frame(self):
        self.frame += 1
        self.step = 0
        
    def save(self, name, image):
        self.step += 1
        mpimg.imsave("%s/%03d_%02d_%s.jpg" % (self.path, self.frame, self.step, name), image)

fancy = fancy_saver()

CALIBRATION_IMAGES = "camera_cal/*.jpg"

calib = calibration(True, fancy)
calib.calibrate(glob.glob(CALIBRATION_IMAGES), 9, 6)

PERSPECTIVE_OFFSET = 450

PLL = [344, 688]
PUL = [620, 445]
PUR = [700, 445]
PLR = [1176, 688]

persp = perspective(PLL, PUL, PUR, PLR, PERSPECTIVE_OFFSET)

mask = np.uint8(mpimg.imread("mask_undist.png")[:,:,0]*255) # couldn't get rid of the alpha channel
#mask = calib.undistort(mask)
#mpimg.imsave("mask_undist.png", mask)



ldec = lane_detector(persp, fancy)

def handle_frame(img):
    if fancy:
        fancy.next_frame()
    undistorted = calib.undistort(img)

    thresh = color_sobel_threshold(undistorted, 3, cv2.COLOR_RGB2HLS, fancy = fancy,
                      mag_threshold = (30, 170),
                      c2_threshold = (105, 255),
                      c3_threshold = (170, 255),
                      )
    masked = cv2.bitwise_and(thresh, mask)


    warped = persp.warp(masked)
    return ldec.process(warped, undistorted)


from moviepy.editor import VideoFileClip

clip1 = VideoFileClip("project_video.mp4").subclip(20,22)
out_clip = clip1.fl_image(handle_frame) 
output = "out.mp4"

out_clip.write_videofile(output, audio=False)


#out_img = ldec.process(warped, undistorted)
#plt.imshow(out_img)
#plt.show()



#perspective ll ul ur lr (255, 677), (597, 448), (687, 448), (1051, 677)

# xabs: 20,140
# hls chan2 145, 255
