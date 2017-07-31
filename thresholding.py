import cv2
import numpy as np
import copy

def processImage(img):
	IMG_WIDTH = img.shape[1]
	IMG_HEIGHT = img.shape[0]
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#img_max = np.amax(img,axis=2)

	#picks out item centers for image#
	ret, thresh = cv2.threshold(img_gray,5,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,0)

	#noise removal
	kernel = np.ones((3,3),np.uint8)
	noiselessimg = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(noiselessimg,2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	sure_fg_copy = copy.deepcopy(sure_fg)

	#get contours
	contours, hierarchy = cv2.findContours( sure_fg_copy,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	#process individual contours
	boxes = []

	MIN_AREA = 100

	for i, contour in enumerate(contours):
		if cv2.contourArea(contour) > MIN_AREA :
			area = cv2.contourArea(contour)

			w = np.sqrt(area)
			h = w
		     
			M = cv2.moments(contour)
			#calculate centroid
			#http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])

			#print IMG_HEIGHT,IMG_WIDTH
			w = min(w, IMG_WIDTH- cx-1)
			h = min(h, IMG_HEIGHT- cy-1)

			box = [int(cx-w/2), int(cy - h/w) , int(w) ,int(h)]

			boxes.append(box)

	#returns list of bounding boxes and threshold image
	return boxes, sure_fg

