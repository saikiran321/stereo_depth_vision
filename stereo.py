import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'c'
def disp(nm):
    cv2.imshow('lol',nm)
    cv2.waitKey(0)
def print_vals(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print [x,y]
        
fig = plt.figure()

fig, ax = plt.subplots()

cv2.namedWindow("Calibration")
vc1 = cv2.VideoCapture(1)
vc2 = cv2.VideoCapture(2)

if vc1.isOpened(): # try to get the first frame
    rval, frame1 = vc1.read()
else:
    rval = False
    print "Cam1 disconected"

if vc2.isOpened(): # try to get the first frame
    rval, frame2 = vc2.read()
else:
    rval = False
    print "Cam2 disconected"
time.sleep(3)
window_size = 6
num_disp = 112
min_disp = -64
kernel = np.ones((8,8),np.uint8)
def nothing(x):
    pass
    
l=20
up=255
cv2.namedWindow("C")
cv2.createTrackbar('l', 'C',0,255,nothing)
cv2.createTrackbar('u', 'C',0,255,nothing)
D=np.array([ -2.1406236596161773e-001, 1.8979559383052200e+000,
       -1.1468160863185207e-002, -8.3620247167440973e-003,
       -5.1832005628890689e+000 ])
CM=np.array([[ 7.1148295989759936e+002, 0., 3.3126394730962716e+002],[ 0.,
       7.1686811893815741e+002, 2.6346019308699692e+002],[ 0., 0., 1. ]])

while rval:
    l = cv2.getTrackbarPos('l','C')
    u = cv2.getTrackbarPos('u','C')
    rval,imgL = vc1.read()
    rval,imgR = vc2.read()
    
    
    both = np.hstack((imgL,imgR))
    #imgL=cv2.fisheye.undistortImage(imgL,CM,D)
    #imgR=cv2.fisheye.undistortImage(imgR,CM,D)
    imgb_L=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgb_R=cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    '''h1, w1 = imgb_L.shape[:2]
    newcamera, roi = cv2.getOptimalNewCameraMatrix(CM, D, (w1,h1), 1,(w1,h1))
    imgb_L = cv2.undistort(imgb_L, CM, D, None, newcamera)
    imgb_R = cv2.undistort(imgb_R, CM, D, None, newcamera)'''
    both = np.hstack((imgb_L,imgb_R))
    
    stereo = cv2.StereoSGBM(
    minDisparity = -39, #min_disp,
    numDisparities = 112,
    SADWindowSize = 5,#window_size,
    uniquenessRatio = 1,
    speckleWindowSize = 150,
    speckleRange = 2,
    disp12MaxDiff = 10,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    fullDP = False
    )
    disparity = stereo.compute(imgb_L, imgb_R)
    norm_image = cv2.normalize(disparity, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    morphology = cv2.morphologyEx(norm_image, cv2.MORPH_OPEN, kernel)
    dd=disparity.astype(np.uint8)
    #ndd = cv2.inRange(dd, l,u)
   
    eroded = cv2.erode(norm_image, np.ones((5,8)))
    dilated = cv2.dilate(eroded, np.ones((25, 25)))
    gray_dd = cv2.inRange(dilated, .9,1.0)
    n_d=gray_dd*255
    (cnts, _) = cv2.findContours(n_d.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    centers=[]
    area=[]
    for c in cnts:
   	cv2.drawContours(dd, [c], -1, (255, 255, 0), 2)
   	max_xy=[[],[]]
   	for a in c:
   	    max_xy[0].append(a[0][0])
   	    max_xy[1].append(a[0][1])
   	max_x=max(max_xy[0])
   	min_x=min(max_xy[0])
   	max_y=max(max_xy[1])
   	min_y=min(max_xy[1])
   	ar=(max_x-min_x)*(max_y-min_y)
   	if ar>100 and ar<5000:
           	center=[(max_x+min_x)/2,(max_y+min_y)/2]
           	centers.append(center)
           	area.append(ar)
           	cv2.circle(dd,tuple( center), 10,255,2 )
    
    
    
    cv2.imshow("Calibration", both)
    cv2.setMouseCallback('C',print_vals)
    cv2.imshow("C", dd)
    
    
    im = ax.imshow(dd,cmap='gray')
    if im:
        im.set_data(dd)
        
    
    ax.format_coord = Formatter(im)
    
    plt.show()
    
    
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        #cv2.imwrite("as.jpg",imgb_L)
        #cv2.imwrite("af.jpg",imgb_R)
        break
cv2.destroyWindow("Calibration")
cv2.destroyWindow("C")


