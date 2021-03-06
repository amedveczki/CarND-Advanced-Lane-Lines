# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.image as mpimg


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
    fancy = None
    
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
            fancy = kwargs[kw]
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
        fancy.save("threshold_%s" % fancy_name, fancy_image)

    return combined


# fixed_max = True
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

# =============================================================================
# mint = 30
# maxt = 170
# thr = 8
# mi = mint
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
