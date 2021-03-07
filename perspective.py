# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.image as mpimg


class perspective:
    def __init__(self, ll, ul, ur, lr, offset, image):
        self.src = np.float32([[ll, ul, ur, lr]])
        self.offset = offset
        self.M = self.IM = None

        # for doc        
        # pts = np.array([ll, ul, ur,lr])
        # pts.reshape((-1, 1, 2))
        # cv2.polylines(image, [pts] , True, (255, 0, 0), 1)
        # mpimg.imsave("before_warp_lines.jpg", image)
        # mpimg.imsave("after_warp_lines.jpg", self.warp(image))

            
    def create_matrix(self, shape):
        self.dst = np.float32([
            [self.offset, shape[0] - 1],
             [self.offset, 0], # UL
             [shape[1] - self.offset, 0], # UR
             [shape[1] - self.offset, shape[0] - 1]
             ])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        _, self.IM = cv2.invert(self.M)
        
    def warp(self, image):
        if self.M is None:
            self.create_matrix(image.shape)
            
        retval = cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        
        return retval

    def unwarp(self, image):
        if self.M is None:
            self.create_matrix(image.shape)
            
        return cv2.warpPerspective(image, self.IM, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


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
