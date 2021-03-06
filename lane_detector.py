# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections

class lane_detector:
    STATE_INIT = 0 # first start or we are completely lost
    STATE_OK = 1 # continue using polynomial+epsilon for searching lane instead of boxes
    STATE_UNCERTAIN = 2 # until MAX_UNCERTAIN_FRAMES we hold onto previous curvature
    MAX_UNCERTAIN_FRAMES = 5 # TODO test

    NUM_SLIDING_WINDOWS = 9
    SLIDING_WINDOW_MARGIN = 70
    MIN_PIXELS_TO_RECENTER_WINDOW = 50

    POLY_WINDOW_MARGIN = 70
    
    MIN_CURVE_RADIUS = 850
    
    KEEP_LAST_FIT = True # Keep last fit
    
    MAX_DRAW_AVERAGE=4 # polynomials are drawn based on this amount of frames
    
    def __init__(self, persp, fancy = None):
        self.state = self.STATE_INIT
        self.fancy = fancy
        self.ploty = None
        self.persp = persp
        self.uncertain_frames = self.MAX_UNCERTAIN_FRAMES
        self.left_fits = collections.deque(maxlen = self.MAX_DRAW_AVERAGE)
        self.right_fits = collections.deque(maxlen = self.MAX_DRAW_AVERAGE)
        
        self.left_fit = self.right_fit = self.left_curverad = self.right_curverad = None
        
    def diff(self,a,b):
        return max(a,b)/min(a,b)
    
    def sanity_check(self, left_curverad, right_curverad,
                     left_fit, right_fit):
        busted = False
        
        if self.left_curverad is not None:
            if self.diff(left_curverad, self.left_curverad) > 0.1:
                busted = True
                
        if not busted and self.right_curverad is not None:
            if self.diff(right_curverad, self.right_curverad) > 0.1:
                busted = True
        
        if left_curverad < self.MIN_CURVE_RADIUS or right_curverad < self.MIN_CURVE_RADIUS:
            busted = True
            
        if busted:
            self.uncertain_frames += 1
            
            if self.uncertain_frames > self.MAX_UNCERTAIN_FRAMES:
                return self.STATE_INIT
            
            return self.STATE_UNCERTAIN
        
        self.uncertain_frames = 0
        return self.STATE_OK
          
    def process(self, warped, undist):
        if self.ploty is None:
            self.ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])

        if self.state == self.STATE_INIT:
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels_histogram(warped)
        else:
            leftx, lefty, rightx, righty, out_img = self.search_around_poly(self.left_fit, self.right_fit, warped)

        left_fit, right_fit, left_fitx, right_fitx = self.fit_poly(leftx, lefty, rightx, righty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)

        left_curverad = (1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**1.5/abs(2*left_fit_cr[0])
        right_curverad = (1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**1.5/abs(2*right_fit_cr[0])

        print("Left: %d meter right: %d meter" % (left_curverad, right_curverad))

        
        self.state = self.sanity_check(left_curverad, right_curverad,
                                       left_fit, right_fit)
        
        if self.fancy:
            ## Visualization ##
            # Colors in the left and right lane regions
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]
            out_img[np.uint16(self.ploty), np.uint16(left_fitx)] = [255, 255, 0]
            out_img[np.uint16(self.ploty), np.uint16(right_fitx)] = [0, 255, 255]

            self.fancy.save("lane_detector_%d_%d" % (left_curverad, right_curverad), out_img)

        def first_or_second_avg(a,b):
            if b is None:
                return a
            
            return (a+b)/2
                    
        if self.state == self.STATE_OK:
            self.left_fit = first_or_second_avg(left_fit, self.left_fit)
            self.right_fit = first_or_second_avg(right_fit, self.right_fit)
            self.left_fits.append(left_fit)
            self.right_fits.append(right_fit)

            self.left_curverad = left_curverad
            self.right_curverad = right_curverad
        else:
            if False and len(self.left_fits) and (not self.KEEP_LAST_FIT or len(self.left_fits) > 1):
                self.left_fits.pop()
                self.right_fits.pop()
                                
            if self.state == self.STATE_INIT:
                self.left_curverad = self.right_curverad = None
        

        projback = self.project_back(undist, warped)

        if self.fancy:
            self.fancy.save("project_back", projback)
            
        return projback

    def search_around_poly(self, left_fit, right_fit, binary_warped):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - self.POLY_WINDOW_MARGIN)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + self.POLY_WINDOW_MARGIN)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - self.POLY_WINDOW_MARGIN)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + self.POLY_WINDOW_MARGIN)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        
        margin = self.POLY_WINDOW_MARGIN
        if self.state  == self.STATE_UNCERTAIN:
            margin *= 0.8
            
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+margin, 
                                  self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+margin, 
                                  self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return leftx, lefty, rightx, righty, result

    def project_back(self, undist, warped):

        if len(self.left_fits) == 0:
            return undist
        
        # Get the average of the last N polys and create new Xs for them
        left_fit = sum(self.left_fits)/len(self.left_fits)
        right_fit = sum(self.right_fits)/len(self.right_fits)
        
        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
        
        
        # Project back the lines
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.persp.unwarp(color_warp) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result


    def find_lane_pixels_histogram(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) if self.fancy else None
        

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        
        # points closer to center are more likely to be lanes
        weight = np.linspace(.5, 1, midpoint)
        leftx_base = np.argmax(histogram[:midpoint] * weight)
        rightx_base = np.argmax(histogram[midpoint:] * weight[::-1]) + midpoint

        # Set height of windows - based on NUM_SLIDING_WINDOWSs above and image shape
        window_height = np.int(binary_warped.shape[0]//self.NUM_SLIDING_WINDOWS)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in NUM_SLIDING_WINDOWSs
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.NUM_SLIDING_WINDOWS):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.SLIDING_WINDOW_MARGIN
            win_xleft_high = leftx_current + self.SLIDING_WINDOW_MARGIN
            win_xright_low = rightx_current - self.SLIDING_WINDOW_MARGIN
            win_xright_high = rightx_current + self.SLIDING_WINDOW_MARGIN
            
            if self.fancy:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > MIN_PIXELS_TO_RECENTER_WINDOW pixels, recenter next window on their mean position
            if len(good_left_inds) > self.MIN_PIXELS_TO_RECENTER_WINDOW:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.MIN_PIXELS_TO_RECENTER_WINDOW:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        self.left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        self.right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
        
        return left_fit, right_fit, self.left_fitx, self.right_fitx
