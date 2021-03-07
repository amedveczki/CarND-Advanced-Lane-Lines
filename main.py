# -*- coding: utf-8 -*-

from calibration import calibration
from thresholds import color_sobel_threshold
from perspective import perspective
from lane_detector import lane_detector
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt

import numpy as np
import cv2


# This is used as a 'picture-debugger' and is given to components if needed.
# Keeps track of counters so things will be in order in the output folder
class fancy_saver:
    def __init__(self, path = "fancy_for_doc2/"):
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


# 1. step - camera calibration
CALIBRATION_IMAGES = "camera_cal/*.jpg"

calib = calibration(True, fancy)
calib.calibrate(glob.glob(CALIBRATION_IMAGES), 9, 6)

# 2. step - prepare perspective transformation matrix (and its inverse)
PERSPECTIVE_OFFSET = 450


PLL = [236,719]
PUL = [589,455]
PUR = [701,455]
PLR = [1179, 719]

#PUL = [585, 460]
#PLL = [203, 720]
#PLR = [1127, 720]
#PUR = [695, 460]

persp = perspective(PLL, PUL, PUR, PLR, PERSPECTIVE_OFFSET, mpimg.imread("persptest.jpg"))




# #==========

# img = mpimg.imread("persptest.jpg")

# minl = 0
# maxl = 1
# step = 1


# curl = minl
# while curl <= maxl:
#      curr = minl
#      while curr <= maxl:
#         persp = perspective(PLL, [PUL[0] + curl, PUL[1]], [PUR[0] + curr, PUR[1]], PLR, PERSPECTIVE_OFFSET)
#         warped = persp.warp(img)
#         mpimg.imsave("warptest/warped_%d_%d.jpg" % (curl, curr), warped)
        
#         if curr == maxl:
#             break
#         curr = min(maxl, curr + step)

#      if curl == maxl:
#         break
#      curl = min(maxl, curl + step)


# import sys
# sys.exit(1)
#==========

# 3. step: load mask which is used to hide the car and things far from the lanes
mask = np.uint8(mpimg.imread("mask.png")[:,:,0]*255) # couldn't get rid of the alpha channel

# 4. Lane detector - this contains lane pixel detection (rectangle/previous polynomial area), polynomial detection
#                    sanity check and projecting the lanes back
ldec = lane_detector(persp, fancy)

# Function which is used to process each frame
def handle_frame(img):
    # Increment the counter in our fancy image saver
    if fancy:
        fancy.next_frame()

    # Use our calibrated camera matrix to undistort the picture
    undistorted = calib.undistort(img)

    # We are using a kernel size of 3, color threshold: HLS, L channel is used (105-255), S channel (170,255).
    # magnitude threshold is 30, 170, which also uses L and S with different weights, but those cannot be set as method parameters.
    thresh = color_sobel_threshold(undistorted, 3, cv2.COLOR_RGB2HLS, fancy = fancy,
                      mag_threshold = (30, 170),
                      c2_threshold = (105, 255),
                      c3_threshold = (170, 255),
                      )

    masked = cv2.bitwise_and(thresh, mask)
    fancy.save("thresh_after_mask", masked)

    warped = persp.warp(masked)
    fancy.save("warped", warped)
    return ldec.process(warped, undistorted)


from moviepy.editor import VideoFileClip

# Save video
clip1 = VideoFileClip("project_video.mp4").subclip(20,20.1)
out_clip = clip1.fl_image(handle_frame) 
output = "out.mp4"

out_clip.write_videofile(output, audio=False)



#perspective ll ul ur lr (255, 677), (597, 448), (687, 448), (1051, 677)

# xabs: 20,140
# hls chan2 145, 255
