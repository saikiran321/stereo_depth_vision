# import the necessary packages
import numpy as np
import cv2
import os 
import time
from math import *

def disp(nm):
    cv2.imshow('lol',nm)
    cv2.waitKey(0)
    
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# lower and upper range red values

# countours processing

cv2.namedWindow("result")
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


def nothing(x):
    pass
# Creating track bar

while rval:
    centers=[]
    area=[]
    centers2=[]
    area2=[]
    rval, frame = vc1.read()
    rval, frame2 = vc2.read()
   
    lower = np.array([77, 0, 0])
    upper = np.array([255, 140, 75])
    #image= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image=frame
    image2=frame2

    mask = cv2.inRange(image,lower,upper)
    mask2 = cv2.inRange(image2,lower,upper)

   

    cv2.imshow('result',mask)
    
    

    #finding contours
    (cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    (cnts2, _) = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print "found %d red shapes" % (len(cnts))
    
    #ct = max(cnts, key = cv2.contourArea)
    #compute the bounding box of the of the paper region and return it
    #print cv2.minAreaRect(ct)
    eroded = cv2.erode(mask, np.ones((5, 5)))
    dilated = cv2.dilate(eroded, np.ones((11, 11)))
    eroded2 = cv2.erode(mask2, np.ones((5, 5)))
    dilated2 = cv2.dilate(eroded2, np.ones((11, 11)))

    for c in cnts:
	cv2.drawContours(image, [c], -1, (255, 255, 0), 2)
	max_xy=[[],[]]
	for a in c:
	    max_xy[0].append(a[0][0])
	    max_xy[1].append(a[0][1])
	max_x=max(max_xy[0])
	min_x=min(max_xy[0])
	max_y=max(max_xy[1])
	min_y=min(max_xy[1])
	ar=(max_x-min_x)*(max_y-min_y)
	center=[(max_x+min_x)/2,(max_y+min_y)/2]
	if ar>2000:
       	    centers.append(center)
       	    area.append(ar)
       	    #print max_x-min_x,max_y-min_y
	    cv2.circle(image,tuple( center), 10,255,2 )
    for c in cnts2:
	cv2.drawContours(image2, [c], -1, (255, 255, 0), 2)
	max_xy2=[[],[]]
	for a in c:
	    max_xy2[0].append(a[0][0])
	    max_xy2[1].append(a[0][1])
	max_x2=max(max_xy2[0])
	min_x2=min(max_xy2[0])
	max_y2=max(max_xy2[1])
	min_y2=min(max_xy2[1])
	ar2=(max_x2-min_x2)*(max_y2-min_y2)
	center2=[(max_x2+min_x2)/2,(max_y2+min_y2)/2]
	if ar2> 2000:
       	    centers2.append(center2)
       	    area2.append(ar2)
       	    #print max_x2-min_x2,max_y2-min_y2
	    cv2.circle(image2,tuple( center2), 10,255,2 )    
    # masking end
    both = np.hstack((image,image2))
    
    try:
        xc=tuple(centers[0])
        xc2=tuple(centers2[0])
        disp_value=sqrt(pow(xc[0]-xc2[0],2)+pow(xc[1]-xc2[1],2))
        dist_val=(744*68)/float(disp_value)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(both,str(dist_val),(100,300), font, 1,(255,0,255),2)
        print "disparity is- %f" %dist_val
    except :
        print " " 
    cv2.imshow("result", dilated)
    cv2.imshow("test",mask)
    cv2.imshow("test1",both)
   
   
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        #cv2.imwrite('sample2.jpg',frame)
        break

cv2.destroyWindow("result")

    
   










