# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle

class calibration:
    CACHE_FILE = "calib.pickle"
    
    def __init__(self, try_load = False, fancy = False):
       self.fancy = fancy
       if try_load:
           try:
               self.mtx, self.dist = pickle.load(open(self.CACHE_FILE, "rb"))
               print("calibration values were loaded from %s" % self.CACHE_FILE)
               return
           except:
               print("could not read cache file %s" % self.CACHE_FILE)
               pass
       self.mtx = None
       self.dist = None
       
       
    def calibrate(self, images_path, corners_x, corners_y):      
        if self.mtx is not None and self.dist is not None:
            print("Cache were used or already initialized, skipping another calibration")
            return
        
        objpoints = [] # world points
        imgpoints = [] # image points
        
        # Fill coordinates 0,0,0...corners_x-1,corners_y-1,0
        # These will be used at every image
        objp = np.zeros((corners_y*corners_x, 3), np.float32)
        objp[:,:2] = np.mgrid[0:corners_x,0:corners_y].T.reshape(-1, 2) 
        
        for path in images_path:
            image = mpimg.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (corners_x, corners_y), None)
            if not ret:
                print("Could not find corners in %s!" % path)
            else:
                imgpoints.append(corners)
                objpoints.append(objp)    
                
                # For doc/debug
                if self.fancy:
                    image = cv2.drawChessboardCorners(image, (corners_x, corners_y), corners, True)
                    mpimg.imsave("fancy/chess_%d.png" % images_path.index(path), image)
    
        # Finally use data to calibrate camera and return matrix
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            with open(self.CACHE_FILE, "wb") as file:
                pickle.dump((mtx, dist), file)
                print("calibration result were written to %s" % self.CACHE_FILE)
            self.mtx = mtx
            self.dist = dist
        else:
            raise Exception("Could not calibrate!")
        
    def undistort(self, image, ):
        if self.mtx is None or self.dist is None:
            raise Exception("Not initialized in undistort")
        undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        
        if self.fancy:
            mpimg.imsave("fancy/undistorted.png", undistorted)
        
        return undistorted