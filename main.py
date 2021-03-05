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
import matplotlib.image as mpimg
import glob



import numpy as np
import cv2
#import matplotlib.image as mpimg

# cv2.COLOR_RGB2HLS, 2,

def abs_sobel_thresh(img, sobel_kernel=3, thresh=(0, 255), orient='x'):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 105, 255 L, c2 (de van benne tul sok vilagos)
# 160, 255 c3
    #mpimg.imsave("fancy/channel0.jpg", grayto3(hls[:,:,0]))
    #mpimg.imsave("fancy/channel1.jpg", grayto3(hls[:,:,1]))
    #mpimg.imsave("fancy/channel2.jpg", grayto3(hls[:,:,2]))
    
    l = hls[:,:,1] * color_thresh(hls, 1, (105, 255)) * 0.6
    s = hls[:,:,2] * color_thresh(hls, 2, (160, 255)) * 1.2
    gray = np.maximum(np.uint8(s), np.uint8(l))
    gray2 = grayto3(gray)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

def grayto3(c):
    return np.dstack((c,c,c))

def color_thresh(img, channel_index, thresh):
    channel = img[:,:,channel_index]   
    # Threshold color channel
    bin_color = np.zeros_like(channel)
    bin_color[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    #binchannel = channel * bin_color
    #fancy_image = np.dstack((binchannel, binchannel, binchannel))
    #mpimg.imsave("fancy/channel_%d_thr%d_%d.jpg" % (channel_index, thresh[0], thresh[1]), fancy_image)

    return bin_color

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # 105, 255 L, c2 (de van benne tul sok vilagos)
# 160, 255 c3
    #mpimg.imsave("fancy/channel0.jpg", grayto3(hls[:,:,0]))
    #mpimg.imsave("fancy/channel1.jpg", grayto3(hls[:,:,1]))
    #mpimg.imsave("fancy/channel2.jpg", grayto3(hls[:,:,2]))
    
    l = hls[:,:,1] * color_thresh(hls, 1, (105, 255)) * 0.6
    s = hls[:,:,2] * color_thresh(hls, 2, (160, 255)) * 1.2
    gray = np.maximum(np.uint8(s), np.uint8(l))
    gray2 = grayto3(gray)
    # mpimg.imsave("fancy/prepgray_thr%d_%d.jpg" % (mag_thresh[0], mag_thresh[1]), gray2)

    
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    maxmag = np.max(magnitude)
    scaled_sobel = np.uint8(255*magnitude/maxmag)
    binary = np.zeros_like(gray)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary
    
def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,2] # not really gray anymore
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    ax = np.abs(sobelx)
    ay = np.abs(sobely)
    
    dirgrad = np.arctan2(ay, ax)
    binary = np.uint8(np.zeros_like(sobelx))
    binary[(dirgrad >= thresh[0]) & (dirgrad <= thresh[1])] = 1
    return binary

def color_sobel_threshold(img, sobel_kernel, color_space, **kwargs):
    # no thresholds by default
    c1_threshold = c2_threshold = c3_threshold = xabs_threshold = mag_threshold = \
        dir_threshold = None
    
    fancy_name = ""
    fancy = False
    
    for kw in kwargs.keys():
        if kw == "c1_threshold":
            c1_threshold = kwargs[kw]
        elif kw == "c2_threshold":
            c2_threshold = kwargs[kw]
        elif kw == "c3_threshold":
            c3_threshold = kwargs[kw]
        elif kw == "xabs_threshold":
            xabs_threshold = kwargs[kw]
        elif kw == "mag_threshold":
            mag_threshold = kwargs[kw]
        elif kw == "dir_threshold":
            dir_threshold = kwargs[kw]
        elif kw == "fancy" and kwargs[kw]:
            fancy = True
            continue
        else:
            raise Exception("Invalid kwarg: %s", kw)
            
        # Dir thresholds are float others are not, remove trailing zeroes at the end after %f
        fancy_name += "_%s_%s_%s" % (kw, ('%3.2f' % kwargs[kw][0]).rstrip('0').rstrip('.'),
                                         ('%3.2f' % kwargs[kw][1]).rstrip('0').rstrip('.'))
   
    # Currently color is & (and), others are | (or'd)
    target_color = cv2.cvtColor(img, color_space)
    bin_c1 = color_thresh(target_color, 0, c1_threshold) if c1_threshold else np.ones_like(img[:,:,0])
    bin_c2 = color_thresh(target_color, 1, c2_threshold) if c2_threshold else np.ones_like(img[:,:,0])
    bin_c3 = color_thresh(target_color, 2, c3_threshold) if c3_threshold else np.ones_like(img[:,:,0])
    
    bin_xabs = abs_sobel_thresh(img, sobel_kernel, xabs_threshold) if xabs_threshold else np.zeros_like(bin_c1)
    bin_mag = mag_thresh(img, sobel_kernel, mag_threshold) if mag_threshold else np.zeros_like(bin_c1)
    bin_dir = dir_thresh(img, sobel_kernel, dir_threshold)  if dir_threshold else np.zeros_like(bin_c1)
 
    combined = np.zeros_like(bin_c1)
    combined[
        (bin_c1 == 1) &
        (bin_c2 == 1) &
        (bin_c3 == 1) |
        (bin_mag == 1) |
        (bin_dir == 1) |
        (bin_xabs == 1)] = 255


    if fancy:
        fancy_image = np.dstack((combined, bin_c2, bin_c3))
        mpimg.imsave("fancy/threshold_%s.jpg" % fancy_name, fancy_image)

    return combined

CALIBRATION_IMAGES = "camera_cal/*.jpg"

calib = calibration(True, True)
calib.calibrate(glob.glob(CALIBRATION_IMAGES), 9, 6)
undistorted = calib.undistort(mpimg.imread("test_images/straight_lines3.jpg"))
# =============================================================================
# 
# target_color = cv2.cvtColor(undistorted, cv2.COLOR_RGB2Luv)
# mpimg.imsave("fancy/threshold_colorconvert_Luv_0_.jpg", np.dstack([target_color[:,:,0], target_color[:,:,0], target_color[:,:,0]]))
# mpimg.imsave("fancy/threshold_colorconvert_Luv_1_.jpg", np.dstack([target_color[:,:,1], target_color[:,:,1], target_color[:,:,1]]))
# mpimg.imsave("fancy/threshold_colorconvert_Luv_2_.jpg", np.dstack([target_color[:,:,2], target_color[:,:,2], target_color[:,:,2]]))
# 
# target_color = cv2.cvtColor(undistorted, cv2.COLOR_RGB2Lab)
# mpimg.imsave("fancy/threshold_colorconvert_Lab_0_.jpg", np.dstack([target_color[:,:,0], target_color[:,:,0], target_color[:,:,0]]))
# mpimg.imsave("fancy/threshold_colorconvert_Lab_1_.jpg", np.dstack([target_color[:,:,1], target_color[:,:,1], target_color[:,:,1]]))
# mpimg.imsave("fancy/threshold_colorconvert_Lab_2_.jpg", np.dstack([target_color[:,:,2], target_color[:,:,2], target_color[:,:,2]]))
# 
# target_color = cv2.cvtColor(undistorted, cv2.COLOR_RGB2YCrCb)
# mpimg.imsave("fancy/threshold_colorconvert_YCrCb_0_.jpg", np.dstack([target_color[:,:,0], target_color[:,:,0], target_color[:,:,0]]))
# mpimg.imsave("fancy/threshold_colorconvert_YCrCb_1_.jpg", np.dstack([target_color[:,:,1], target_color[:,:,1], target_color[:,:,1]]))
# mpimg.imsave("fancy/threshold_colorconvert_YCrCb_2_.jpg", np.dstack([target_color[:,:,2], target_color[:,:,2], target_color[:,:,2]]))
# 
# target_color = cv2.cvtColor(undistorted, cv2.COLOR_RGB2YUV)
# mpimg.imsave("fancy/threshold_colorconvert_IYUV_0_.jpg", np.dstack([target_color[:,:,0], target_color[:,:,0], target_color[:,:,0]]))
# mpimg.imsave("fancy/threshold_colorconvert_IYUV_1_.jpg", np.dstack([target_color[:,:,1], target_color[:,:,1], target_color[:,:,1]]))
# mpimg.imsave("fancy/threshold_colorconvert_IYUV_2_.jpg", np.dstack([target_color[:,:,2], target_color[:,:,2], target_color[:,:,2]]))
# 
#target_color = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
#mpimg.imsave("fancy/threshold_colorconvert_HLS_0_.jpg", np.dstack([target_color[:,:,0], target_color[:,:,0], target_color[:,:,0]]))
#mpimg.imsave("fancy/threshold_colorconvert_HLS_1_.jpg", np.dstack([target_color[:,:,1], target_color[:,:,1], target_color[:,:,1]]))
#mpimg.imsave("fancy/threshold_colorconvert_HLS_2_.jpg", np.dstack([target_color[:,:,2], target_color[:,:,2], target_color[:,:,2]]))

#xx = mpimg.imsave("fancy/threshold_colorconvert_HLS_12_.jpg", np.dstack([np.zeros_like(target_color[:,:,2]), target_color[:,:,1], target_color[:,:,2]]))
# =============================================================================

class perspective:
    def __init__(self, ll, ul, ur, lr, offset, img):
        src = np.float32([[ll, ul, ur, lr]])
        dst = np.float32([
            [offset, img.shape[0] - 1],
             [offset, 0], # UL
             [img.shape[1] - offset, 0], # UR
             [img.shape[1] - offset, img.shape[0] - 1]
             ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        
    def unwarp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


PERSPECTIVE_OFFSET = 250

PLL = [344, 688]
PUL = [620, 445]
PUR = [700, 445]
PLR = [1176, 688]


minl = -1
maxl = 1
step = 1


curl = minl
while curl <= maxl:
    curr = minl
    while curr <= maxl:
        persp = perspective(PLL, [PUL[0] + curl, PUL[1]], [PUR[0] + curr, PUR[1]], PLR, PERSPECTIVE_OFFSET, undistorted)
#        persp = perspective([PLL[0]+curl, PLL[1]], PUL, PUR, [PLR[0]+curr, PLR[1]], PERSPECTIVE_OFFSET, undistorted)
        warped = persp.unwarp(undistorted)
        mpimg.imsave("fancy/warped_%d_%d.jpg" % (curl, curr), warped)
        
        if curr == maxl:
            break
        curr = min(maxl, curr + step)

    if curl == maxl:
        break
    curl = min(maxl, curl + step)



mint = 30
maxt = 170
thr = 8
mi = mint

# 105, 255 L, c2 (de van benne tul sok vilagos)
# 160, 255 c3

# 150, 255 C2 maghoz?
fixed_max = True

# =============================================================================
# while mi < maxt:
#     ma = maxt if fixed_max else min(maxt, mi + thr)
#     
#     while ma <= maxt:
#         color_sobel_threshold(undistorted, 3, cv2.COLOR_RGB2HLS, fancy = True, c3_threshold = (mi, ma))
#        
#         if ma == maxt:
#             break
#         
#         ma = min(maxt, ma + thr)
# 
#     mi += thr
# # 30, 170 mag
# =============================================================================
color_sobel_threshold(undistorted, 3, cv2.COLOR_RGB2HLS, fancy = True,
                      mag_threshold = (30, 170),
                      c2_threshold = (105, 255),
                      c3_threshold = (170, 255),
                      )


#perspective ll ul ur lr (255, 677), (597, 448), (687, 448), (1051, 677)

# xabs: 20,140
# hls chan2 145, 255