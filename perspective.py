# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.image as mpimg


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
        
    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


# =============================================================================
# minl = -1
# maxl = 1
# step = 1
# 
# 
# curl = minl
# while curl <= maxl:
#     curr = minl
#     while curr <= maxl:
#         persp = perspective(PLL, [PUL[0] + curl, PUL[1]], [PUR[0] + curr, PUR[1]], PLR, PERSPECTIVE_OFFSET, undistorted)
# #        persp = perspective([PLL[0]+curl, PLL[1]], PUL, PUR, [PLR[0]+curr, PLR[1]], PERSPECTIVE_OFFSET, undistorted)
#         warped = persp.unwarp(undistorted)
#         mpimg.imsave("fancy/warped_%d_%d.jpg" % (curl, curr), warped)
#         
#         if curr == maxl:
#             break
#         curr = min(maxl, curr + step)
# 
#     if curl == maxl:
#         break
#     curl = min(maxl, curl + step)
# 
# =============================================================================
