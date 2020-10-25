#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import getopt
import time
import json

# Number of pictures to be used in the calibration process
NUM_PICTURES = 10

# Sample period for image capture display one frame every IMG_SAMPLING_PERIOD seconds
# mishow over SSH is slow and delay the frames processing. Therefore we display a
# frame only every IMG_SAMPLING_PERIOD seconds
IMG_SAMPLING_PERIOD = 4.0

# Chessboard dimensions
CHESSBOARD = (7,5)

DEFAULT_OUT_FILE = 'calibration.json'

OPTIONS = '[--file <calibration_file>] [--test]'

cam = cv2.VideoCapture(0)

"""
Process input arguments.
Options:
    --test: test only mode. Reads calibration file and displays normal and undistorded frames
    --file <calibration_file>: calibration file
"""
def process_argv(argv):
    calibration_file = DEFAULT_OUT_FILE
    test = False
    try:
        opts, args = getopt.getopt(argv[1:], "hf:t", ["file=","test"])
    except getopt.GetoptError:
        print('{0} {1}'.format(argv[0], OPTIONS))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('{0} {1}'.format(argv[0], OPTIONS))
            sys.exit()
        elif opt in ("-f", "--file"):
            calibration_file = arg
        elif opt in ("-t", "--test"):
            print('Test only mode ...')
            test = True

    return test, calibration_file

"""
Store camera calibration parameters in a JSON file.
"""
def store_calibration_param(mtx, dist, calibration_file = DEFAULT_OUT_FILE):
    data = {"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()}
    
    with open(calibration_file, "w") as f:
        json.dump(data, f)

    print('Calibration parameters stored in: {0}'.format(calibration_file))


def load_calibration_param(calibration_file = DEFAULT_OUT_FILE):
    
    print('Reading calibration parameters from: {0}'.format(calibration_file))

    with open(calibration_file, "r") as f:
        data = json.load(f)
        return np.array(data['camera_matrix']), np.array(data['dist_coeff'])


"""
Capture and display undistorded frames using the calibration parameters.
To be called after the calibration process for testing purposes.
"""
def test_camera(mtx, dist):

    cv2.namedWindow("Normal")
    cv2.namedWindow("Undistorded")

    last_frame_time = 0

    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame from camera.")
        cv2.destroyAllWindows()
        return
    else:
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
        x,y,w,h = roi

    while True:
        cur_time = time.time()
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

            # Display a frame only every IMG_SAMPLING_PERIOD seconds
        if cur_time - last_frame_time > IMG_SAMPLING_PERIOD:
            last_frame_time = cur_time

            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            dst = dst[y:y+h, x:x+w]
            cv2.imshow("Normal", img)
            cv2.imshow("Undistorded", dst)
    
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    
    cv2.destroyAllWindows()

"""
Performs camera calibration and returns the calibration matrix and dist
"""
def calibrate():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

    cv2.namedWindow("Preview")

    img_counter = 0
    last_frame_time = 0

    print("Grabbing chessboard pictures. Press space to grab a picture.")

    while img_counter < NUM_PICTURES:
        
        cur_time = time.time()
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Display a frame only every IMG_SAMPLING_PERIOD seconds
        if cur_time - last_frame_time > IMG_SAMPLING_PERIOD:
            last_frame_time = cur_time
            cv2.imshow("Preview", img)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_counter += 1
            print('Capuring image {0} ...'.format(img_counter))
            
            # print("Saving image ...")
            # img_name = "chessboard_{}.png".format(img_counter)
            # cv2.imwrite(img_name, img)
            # print("{} written!".format(img_name))

            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray_img, CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
                cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of chess board
            """
            if ret == True:
                print('Found chessboard corners.')
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners = cv2.cornerSubPix(gray_img, corners, (11,11), (-1,-1), criteria)
                
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHESSBOARD, corners, ret)
                
                # wait IMG_SAMPLING_PERIOD seconds before showing next frame
                last_frame_time = time.time()
                cv2.imshow("Preview", img)
                
            else:
                print('Unable to find chessboard corners.')

    cv2.destroyAllWindows()

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    if len(objpoints) == NUM_PICTURES:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1],None,None)
        return True, mtx, dist
    else:
        return False, np.array([]), np.array([])

def destroy():
    cam.release()
    cv2.destroyAllWindows()

def main(test_mode, calibration_file):
    ret = True
    if not test_mode:
        ret, mtx, dist = calibrate()
        if ret:
            store_calibration_param(mtx, dist, calibration_file=calibration_file)
    else:
        mtx, dist = load_calibration_param(calibration_file=calibration_file)

    if ret:
        test_camera(mtx, dist)

if __name__ == '__main__':
    try:
        test_mode, calibration_file = process_argv(sys.argv)
        main(test_mode, calibration_file)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Exiting ...')
    finally:
        destroy()