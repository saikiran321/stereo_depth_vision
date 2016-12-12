
import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*5,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpointsL = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
objpointsR = []
imgpointsR = []

images = glob.glob('l_*.jpg')

for fname in images:
    img = cv2.imread(fname)
    grayL = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, cornersL = cv2.findChessboardCorners(grayL, (7,5),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsL.append(objp)

        cv2.cornerSubPix(grayL,cornersL,(7,5),(-1,-1),criteria)
        imgpointsL.append(cornersL)


images = glob.glob('r_*.jpg')

for fname in images:
    img = cv2.imread(fname)
    grayR = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, cornersR = cv2.findChessboardCorners(grayR, (7,5),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsR.append(objp)

        cv2.cornerSubPix(grayR,cornersR,(7,5),(-1,-1),criteria)
        for x in cornersR:
            cv2.circle(img,tuple(x), 10,255,1 )
        cv2.imshow('disparity', img)

        cv2.waitKey(1)
        imgpointsR.append(cornersR)

cameraMatrix1 =None
cameraMatrix2 = None
distCoeffs1 = None
distCoeffs2 = None
R =None
T = None
E = None
F = None
w=640
h=480
#retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpointsL,imgpointsL, imgpointsR,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(w,h), R, T, E, F)
retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpointsL, imgpointsL, imgpointsR, (640,480))