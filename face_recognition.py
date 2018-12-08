'''
Sarath Kumar
sarathsmartzs[at]gmail[dot]com
SACET, Chirala

Usage: face_recognition.py <full/path/to/root/images/folder> <full/path/to/images/folder/to/predict>
Takes two arguments:
	1. Input folder which contains sub-folders of subjects/ persons.
		There should be images saved in subfolders which are used to train.
	2. Path to a folder consists of images for which we are gonna predict subject.
		Note: Images in here should be cropped and have only faces of subjects.
'''

import cv2
import numpy as np
from os import listdir
import sys, time
import face_recognition
import dlib

def get_images(path, size):
	'''
	path: path to a folder which contains subfolders of for each subject/person
		which in turn cotains pictures of subjects/persons.

	size: a tuple to resize images.
		Ex- (256, 256)
	'''
	sub= 0
	images, labels= [], []
	people= []

	for subdir in listdir(path):
		for image in listdir(path+ "/"+ subdir):
			#print(subdir, images)
			img= cv2.imread(path+"/"+subdir+"/"+image, cv2.IMREAD_GRAYSCALE)
			img= cv2.resize(img, size)

			images.append(np.asarray(img, dtype= np.uint8))
			#get the label for the image
			nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
			for (x,y,w,h) in faces:
				images.append(image[y:y + h, x: x + w])
				labels.append(nbr)
			#labels.append(sub)

			#cv2.imshow("win", img)
			#cv2.waitKey(10)

		people.append(subdir)
		sub+= 1

	return [images, labels, people]



if __name__== "__main__":
	if len(sys.argv)!= 3:
		print("Wrong number of arguments! See the usage.\n")
		print("Usage: face_recognition.py <full/path/to/root/images/folder> <full/path/to/images/folder/to/predict>")
		sys.exit()

	[images, labels, people]= get_images(sys.argv[1], (256, 256))
	#print([images, labels])

	labels= np.asarray(labels, dtype= np.int32)

	# initializing eigen_model and training
	print("Initializing eigen FaceRecognizer and training...")
	sttime= time.clock()
	eigen_model= cv2.createEigenFaceRecognizer()
	eigen_model.train(images, labels)
	print("\tSuccessfully completed training in "+ str(time.clock()- sttime)+ " Secs!")

	# starting to predict subject/ person
	for image_name in listdir(sys.argv[2]):
		try:
			pre_image= cv2.imread(sys.argv[2]+ "/"+ image_name, cv2.IMREAD_GRAYSCALE)
			pre_image= cv2.resize(pre_image, (256, 256))
		except:
			print("Couldn't read image. Please check the path to image file.")
			sys.exit()

		[predicted_label, predicted_conf]= eigen_model.predict(np.asarray(pre_image))
		print("Predicted person in the image "+ image_name+ " : "+ people[predicted_label])
