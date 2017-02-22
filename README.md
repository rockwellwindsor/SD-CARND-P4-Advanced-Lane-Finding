# Advanced Lane Finding
### Udacity Self-Driving Car Nanodegree - Project 4 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
### Objective

The main objective of this project is to write a software pipeline to identify the lane boundaries in a video.  With the main output or product being the creation of a detailed writeup of the project.   

Steps of this project are the following:
* 1) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* 2) Apply a distortion correction to raw images.
* 3) Use color transforms, gradients, etc., to create a thresholded binary image.
* 4) Apply a perspective transform to rectify binary image ("birds-eye view").
* 5) Detect lane pixels and fit to find the lane boundary.
* 6) Determine the curvature of the lane and vehicle position with respect to center.
* 7) Warp the detected lane boundaries back onto the original image.
* 8) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Steps
---

* 1) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    * Camera callibration

        This code was taken from the lesson on stage 9 <strong>"Finding Corners"</strong>

    ````
    # Calibrate Camera
    # Draw lines on the chessboard, taken from slide 9. Finding Corners from the lesson.
    objpoints = [] # 3d points in real world space.  Should all be the same. (See 10 - Calibrating Camera)
    imgpoints = [] # 2d points in image plane.

    objp = np.zeros((6*9,3), np.float32) # Needs to be 9, not 8 like in the video.
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Glob api reads in all the images, must have consistant file name.
    images = glob.glob('camera_cal/calibration*.jpg')

    # Loop through each calibration image
    for image, fname in enumerate(images):
        
        img = cv2.imread(fname)
        gray = grayscale(img)

        # Find the chessboard corners.  This allows us to get coordinates of corners on distorted image.
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points and image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
            ax1.imshow(cv2.cvtColor(mpimg.imread(fname), cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image', fontsize=18)
            ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax2.set_title('With Corners', fontsize=18)
    ````
    Produces the undistorted image below.

<p align="center"><img src="./images/chessboards-lines.png" alt="End result"  /></p>

* 2) Apply a distortion correction to raw images.
    * Distortion correction 

        This code comes from stage 11 <strong>"Correcting for Distortion"</strong>

    ````

    # Loop through and display each undistorted image.
    for image, fname in enumerate(images):
        
        img = cv2.imread(fname)
        
        undistorted = cal_undistort(img, objpoints, imgpoints)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undistorted)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    ````

    Produces the undistorted image below.

<p align="center"><img src="./images/chessboards-dist-undist.png" alt="End result"  /></p>

* 3) Use color transforms, gradients, etc., to create a thresholded binary image.
    * Color transforms
    * Gradient

* 4) Apply a perspective transform to rectify binary image ("birds-eye view").
    * Perspective transform

    This code comes from stage 17, <strong>"Undistort and Transform"</strong>

    ````

    t_image = mpimg.imread('test_images/test5.jpg')

    w_image = warp(t_image)

    p, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

    ax1.set_title('Original Image', fontsize=40)
    ax1.imshow(t_image)

    ax2.set_title('Warped Result', fontsize=40)
    ax2.imshow(w_image)

    ````

    Produces the undistorted image below.

<p align="center"><img src="./images/perspective-transform.png" alt="End result"  /></p>

* 5) Detect lane pixels and fit to find the lane boundary.
    * Fit
* 6) Determine the curvature of the lane and vehicle position with respect to center.
    * Curvature
    * Vehicle position
* 7) Warp the detected lane boundaries back onto the original image.
    * Warp
* 8) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    * Output display of lane boundries
    * Generate numerical estimates
        * Lane curvature
        * Vehicle position

### Acknowledgments
---
