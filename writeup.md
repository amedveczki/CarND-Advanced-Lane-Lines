## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[undistort_1]: ./output_images/undistort_chess.gif "Chess distort/undistort"
[undistort_2]: ./output_images/undistort_road.gif "Road distort/undistort"
[threshold_l]: ./output_images/001_03_c2.jpg "HLS L channel result"
[threshold_s]: ./output_images/001_04_c3.jpg "HLS S channel result"
[threshold_mag]: ./output_images/001_05_bin_mag.jpg "Gradient magnitude result"
[threshold_final]: ./output_images/001_06_threshold__mag_threshold_30_170_c2_threshold_105_255_c3_threshold_170_255.jpg "Final threshold"
[before_warp_lines]: ./output_images/before_warp_lines.jpg "Warp before"
[after_warp_lines]: ./output_images/after_warp_lines.jpg "Warp after"
[mask_animated]: ./output_images/mask_animated.gif "Animated mask"
[lane_detect_window]: ./output_images/001_09_lane_detector_952_792.jpg "Lane detection windows"
[poly_curves]: ./output_images/002_09_lane_detector_987_816.jpg "Using previous polyline"
[final]: ./output_images/001_10_project_back.jpg "Project back"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is similar to the one in the lessons. The whole calibration is put into `calibration.py`. For convinience, it also caches the output it makes (through `pickle`), so it will be quick in subsequent runs.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][undistort_1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

For per-frame logic the code is mostly in `main.py`. It uses the mentioned `calibration.py` - after the calibration matrix has been finished, it is used for every frame (method `calibration.undistort`) to remove camera distortion from the images.

See example below.

![alt text][undistort_2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The related source code can be found in `thresholds.py`. It has much unused code currently - the method which is used by the actual code (`color_sobel_threshold`) is configurable via "kwargs" - every threshold can be enabled, disabled, color threshold can be used per-channel with any given color space.

Currently in `main.py`HLS  L/S channels are used in color threshold, along with gradient magnitude threshold (which also uses L/S channels, weighted, 0.6 and 1.2 weights respectively.) 

HLS L channel result:

![L channel][threshold_l]

HLS S channel result:

![S channel][threshold_s]

Magnitude gradient based on L,S channels, weighted:

![Magnitude gradient][threshold_mag]

The final result is {all color thresholds, "binary and"} | magnitude threshold.

![Final result][threshold_final]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in the file `perspective.py`. It can be initialized with the source coordinates, along with an offset (the matrix itself is computed at its first usage - so the class `perspective` can be passed as initialization to an other class without knowing the target shape).

Inverse transformation (`unwarp()`) is also done with this class.

The hard-coded source points were initially made up with putting lines on the picture by eye and using the intersection points but that wasn't perfect, so I ended up rendering the possible warped pictures using +/- 1..5 pixels away (X direction) from the original, and I chose the perfect one.  

| Point       |  Source   |           Destination           |
| ----------- | :-------: | :-----------------------------: |
| Lower left  |  236,719  |       offset, shape[0]-1        |
| Upper left  |  589,455  |            offset, 0            |
| Upper right |  701,455  |      shape[1] - offset, 0       |
| Lower right | 1179, 719 | shape[1] - offset, shape[0] - 1 |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Before warp][before_warp_lines]

![after warp lines][after_warp_lines]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A mask is applied to the output of thresholds which removes the car and some far-from-lane parts which could sometimes otherwise affect the lane lines.

![mask][mask_animated]

Now the magic happens in `lane_detector.py`. It cries for refactor now, but roughly this is what it does (`process()`):

1. We try to find the pixels for the left and right lanes.
   1. At start (and when we realize we are completely lost) we are using `find_lane_pixels_histogram()` - we build a histogram on the bottom half of the picture, and search for the initial X coordinates at the bottom for each lane.
      The histogram is weighted between 0.5 and 1 based on how far it is from the middlepoint as at some frames there were some vertical lines unrelated to the actual lanes which were detected as lane lines. The weighing (and masking) solves it most of the time.
   2. As you can see below, there are "windows" put on top of each other trying to track the pixels down for the lanes - their center position is affected by the already found pixels before, if enough pixels are found the next window has its center position at their average X values. 
   3. (If it's not the first run and we have trust in the previous lanes, we are using those (or the average of the last 2) with a given margin, see the second picture below. This is one in `search_around_poly()` )
2. Based on the pixels in each we try to fit a polynomial to each one using numpy's great `polyfit()` method. (`fit_poly()` in `lane_detector.py`) 

![Lane detection with windows][lane_detect_window]

![Curves][poly_curves]

3. We have a sanity check - after computing curvature if we have previous values (and they are less than a given "maximum" which would indicate it's more a straight line), we compare it to the previous values, if it differs more than 30%, we consider it as "busted", a bad frame. If it's less than a given limit, we also mark it as bad. `lane_detector.sanity_check()` is responsible for this task.
   After a certain amount of bad frames (currently 2) we start from scratch using the sliding window method.

4. We are using the average of the last 3 frames for drawing (but we consider only the good ones - and we keep the last good until there is a new one). (And now I realize it isn't dropped if we have a new one, this could be fixed.)

   

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I stuck with the values given in the lesson for x/y meter/pixel as using my method was unsuccessful (one line strip should be 10 feet, this can be measured in the projected image, and the distance between left and right lanes can be also measured).

So radius is computed based on the equation in the lessons, found in `calc_radius_and_carpos()`. The car position computed as the following:


$$
\frac{middlepoint_x - left\_lane\_bottom_x}{difference\_of\_bottom\_lane\_pixels}
$$
This gives a [0..1] value where the car is. This can be multiplied with the lane width (3.7) and we get the camera distance from the left lane in meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The drawn lanes are projected back in `lane_detector.project_back()` (which uses the `unwrap()` method of `perspective`).

The text is overlayed in the last few lines of `lane_detector.process()`.

![Final image][final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My perspective transformation was not perfect at first and even though it was perfect for the picture I did it (likely it was distorted by mistake?), radius sanity check, and other thresholds needed to be re-done as it was just not enough. Fortunately I realized there was a problem with the perspective.

The thresholding took too much time; I generated many many pictures with many combinations. Instead I should have tried to do it with some sliders using a GUI (a colleague has recommended an easy to use solution since then: pyqtgraph ).

I should have done some tests after each modification though since the detection depends on the past it's not so easy anymore.

Sanity check should include much more, like center of camera position.

There are still places where a few wrong pixels are driving the polynomial way off the real lane.

....and I could have done much more, but I'm way behind schedule.