import csv
from random import shuffle, sample
import numpy as np
import cv2

print "imported BakeryDataset"
class BakeryDataset():

	def __init__(self):

		self.filelist = []
		self.labelsDict = {"bananabread":0, "cinnamonroll":1, "croissant":2, "hotcross":3}
		self.numclasses = len(self.labelsDict)
		self.datapath = '/Users/RoyChan/YOLO_tensorflow/dataset/resized448/'


		counter = 0
		#import filenames and labels to filelist
		with open('/Users/RoyChan/YOLO_tensorflow/dataset/labels.csv', 'rU') as csvfile:
			reader = csv.reader(csvfile,  dialect=csv.excel_tab, delimiter=',')
			for row in reader:
				self.filelist.append(row)
				counter += 1
				
		self.total_images = counter

		#randomize filelist order
		shuffle(self.filelist)

	def pickSample(self, sample_size):

		#pick a sample from filelist, returns [SAMPLE_SIZE,448,448,3] array
		pickedSample = sample(self.filelist,sample_size)

		#stack sample into [sample_size,448,448,3] array
		imgArray = np.zeros((sample_size,448,448,3))
		#labels array of size [sample_size, 4]
		labels = np.zeros((sample_size,self.numclasses))

		for i in range(sample_size):

			#image
			img = cv2.imread(self.datapath + pickedSample[i][0])
			imgArray[i,:,:,:] = img

			#labels
			objectName = pickedSample[i][1]
			onehotPosition = self.labelsDict[objectName]
			labels[i,onehotPosition] = 1 


		return imgArray, labels

if __name__ == "__main__":

	x = BakeryDataset()
	imgs, y = x.pickSample(25)
	print y

