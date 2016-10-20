#! /usr/bin/env python3
# Module pointOperators.py


#################################################
#												#
#	Python functions for point pointOperators 	#
#												#
#	Author: Robert Thomas						#
#	Username: rjst20							#
#												#
#################################################

from imageImport import *

# Allow for plotting histogram graphs
import matplotlib.pyplot as plt

def negImage(imageStruct):
	# Function to invert the greyscale image
	initialImage = imageStruct.image
	newImage = 255-initialImage 
	return newImage

def greyLvlSlice(imageStruct, lMin, lMax):
	# Duplicate image to new structure so that it is possible to animate easily

	newImage = imageStruct.image

	lowerSubset = newImage <=lMin
	midSubset = (newImage > lMin) & (newImage <= lMax)
	upperSubset = newImage > lMax

	newImage[lowerSubset] = 0
	newImage[midSubset] = 255
	newImage[upperSubset] = 0

	return newImage

def greyLvlSliceDisp(imageStruct):
	for i in range (1,21):
		animSlice = imageStruct.duplicate()
		dispImage = greyLvlSlice(animSlice, (5*i), (10*i))
		dispImage.showImage()

def makeHistogram(imageStruct):
	# Function to create a histogram of an image's greyscale levels using the formula:
	# p(r) = n / MN

	# Find the total number of elements in the image
	imageM, imageN = imageStruct.imageSize
	imageMN = imageM * imageN

	# Initialise an array for counting the occurences of each grey value
	nkCount = np.zeros(256, 'uint8')

	for i in range(imageM):
		for j in range(imageN):
			nkCount[imageStruct.image[i,j]] = nkCount[imageStruct.image[i,j]] + 1


	# Calculate probability of occurence
	probRk = nkCount / imageMN

	plt.plot(probRk)
	plt.xlim(0,255)
	plt.show()

	return nkCount, imageMN

def normaliseHistogram(imageStruct, nMin, nMax):
	# Function to utilise contrast stretching for a given image using the formula:
	# f(x,y) = [f(x,y) - Omin] x (Nmax-Nmin)/(Omax-Omin) + Nmin

	# Ensure that the boundaries are floats so that the conversion ratio is a float
	nMin = float(nMin)
	nMax = float(nMax)

	# Find Omax and Omin Values
	oMin = np.amin(imageStruct.image)
	oMax = np.amax(imageStruct.image)

	# Generate the conversion ratio to reduce the amount of divisions required
	conversionRatio = ((nMax - nMin)/(oMax-oMin))


	imageM, imageN = imageStruct.imageSize()
	newImage = np.zeros([imageM, imageN],'uint8')

	for i in range(imageM):
		for j in range(imageN):
			newImage[i,j] = int((imageStruct.image[i,j] - oMin) * conversionRatio + nMin)

	return blankImage






















	










#a = Image('NZjers1.png')
a = Image('foetus.png')

#makeHistogram(a)
#a.showImage()
normaliseHistogram(a, 0, 255)


a.showImage()
a.negImage()
a.showImage()
