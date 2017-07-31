##Resizes images in directory to 448x448 and save csv file containing labels which are folder names

import os
import cv2
import csv

directory = './'
data = []

for path, subdirs, files in os.walk(directory):
    for name in files:
    	if str(name).endswith(".JPEG") and path[2::] != "resized448":
	        print os.path.join(path, name)
	        print path

	        #process image and save
	        img = cv2.imread(os.path.join(path, name))
	        img = cv2.resize(img,(448,448))
	        cv2.imwrite('./resized448/' + str(name), img)
	        #cv2.imshow("test",img)
	        #cv2.waitKey(0)
	        data.append([str(name),path[2::]])

#write filename and labels to csv
with open('./labels.csv','w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerows(data)