#! /usr/bin/env python3
# Module imageImport.py


#################################################
#												#
#	Python functions for importing image files 	#
#												#
#	Author: Robert Thomas						#
#	Username: rjst20							#
#												#
#################################################


# OpenCV used for loading images
from cv2 import *


# Modules below used for listing available images in the current directory
from os import getcwd
from os.path import join, split
from glob import glob

# Modules for function termination and writing to file
from sys import exit
import sys
import numpy as np

# Allow for plotting histogram graphs
import matplotlib.pyplot as plt

class Image:
	# Define a class that contains an image and its associated name
	'Common class for an image'
	"""
	Attributes:
		name: A string containing the name of the original image
		image: An array of the individual pixels of the image
		transforms: A list of all performed transforms done on the image
	"""
	def __init__(self, *args):
		# Constructor for the class

		# Generate a list of files within the current directory for validation
		listOfFiles = self._listFiles()

		# No transforms have been yet done on the image
		self.transforms = []

		if (0 == len(args)): # Validate that there is an argument to import, if not generate an empty image
			[self.image, self.name] = [ [], "" ]
		else: # If a string is provided check that it exists within the current directory
			args = args[0]# Select only the first argument provided if it exists
			if (args) in listOfFiles: # If filename is present within the current directory import the file and return it
				imageImported = self._importPic(args)
				__printSpacer__('Image file ' + args + ' succesfully loaded')
				[self.image, self.name] = imageImported, args
			elif 'empty' == args:
				self.image = []
				self.name = 'emptyImage'
			else: # If file does not exist within the directory exit the scripts
				__printSpacer__(args + ': file not found')
				self._printFileList(listOfFiles)
				exit()

	##########################################
	##										##
	## Methods (Internal):					##
	##										##
	##########################################

	def _listFiles(*args):
		#Function to list available files for processing in the current directory
		# Define the current working directory
		cwd = getcwd() 
		# Create a glob-able parser
		pngFilter = join(cwd,'*.png')
		# Extract an array of all pngs in the current working directory
		pngGlob = glob(pngFilter)

		# Initialise an empty array to contain all valid png image files in the directory
		pngFiles = []

		# For each image detected in the current working directory remove the directory path and 
		# add file to pngFiles array
		for image in pngGlob:
			_ , fileName = split(image)
			pngFiles.append(fileName)

		# return the populated array of file names
		return pngFiles

	def _importPic(*args):
		# Function to import an image corresponding to a pre-validated string

		# OpenCV's function 'imread' brings in a grayscale image
		# 1 = color image without alpha channel
		# 0 = grayscale image
		# -1 = unchanged image including alpha channel
		imageImported = imread(args[1],0)

		# Return imported image
		return imageImported

	def _printFileList(_,listOfFiles):
		# Function to print the files within the list
		__printSpacer__()
		print('No input file selected, please enter one of the following files as a function input:')
		print('')
		for entry in listOfFiles:
			print(entry)
		__printSpacer__()


	##########################################
	##										##
	## Methods (External):					##
	##										##
	##########################################

	def imageSize(self):
		imageM, imageN = self.image.shape
		return imageM, imageN

	def updateImage(self, newImage, transform):
		# Update the image after a new transform
		self.image = newImage
		self.transforms.append(transform)

	def duplicate(self):
		# Function to create a duplicate of the current structure
		newStruct = Image('empty')
		# Hard assigns to stop altering original
		imageM, imageN = self.imageSize()

		newStruct.image = np.add(np.zeros([imageM, imageN],'uint8'), self.image)
		newStruct.name = 'Duplicate' + self.name
		newStruct.transforms = self.transforms + ['Duplicate']
		return newStruct

	def mask(self, mask):
		# Function to apply a mask provided to the image
		print('Applying mask')
		self.image = np.minimum(self.image, mask.image)
		print('Mask done')

	def overlay(self, imageOver, mask):
		# Function to overlay two images of the same size using a defined mask
		# Apply mask to the base image
		self.mask(mask)
		# Invert the mask
		mask.POnegImage()
		# Apply the inverted mask to the image to overlay
		imageOver.mask(mask)
		# Combine the two masked images
		self.image = self.image + imageOver.image


	##########################################
	##										##
	## Methods (Outputs)					##
	##										##
	##########################################
	def showImage(self):
		# Shows the contained image until a key is pressed
		imshow(self.name, self.image)
		waitKey(0)
		destroyAllWindows()

	def saveImage(self):
		# Function to save an image to file after processing
		imwrite(self.name + "_".join(self.transforms) + ".png", self.image)

	def histogram(self): 
		# Function to create a histogram of an image's greyscale levels using the formula:
		# p(r) = n / MN

		# Find the total number of elements in the image
		imageM, imageN = self.imageSize()
		imageMN = imageM * imageN *1.0
		print(imageMN)

		# Initialise an array for counting the occurences of each grey value
		nkCount = np.zeros(256, 'uint8')

		for i in range(imageM):
			for j in range(imageN):
				nkCount[self.image[i,j]] = nkCount[self.image[i,j]] + 1.0


		# Calculate probability of occurence
		probRk = nkCount / imageMN

		plt.plot(probRk)
		plt.xlim(0,255)
		plt.show()
	##########################################
	##										##
	## Methods (Point Operator Filters)		##
	##										##
	##########################################
	def POnormalise(self, nMin, nMax):
		# Function to utilise contrast stretching for a given image using the formula:
		# f(x,y) = [f(x,y) - Omin] x (Nmax-Nmin)/(Omax-Omin) + Nmin

		# Ensure that the boundaries are floats so that the conversion ratio is a float
		nMin = float(nMin)
		nMax = float(nMax)

		# Find Omax and Omin Values
		oMin = np.amin(self.image)
		oMax = np.amax(self.image)

		# Generate the conversion ratio to reduce the amount of divisions required
		conversionRatio = ((nMax - nMin)/(oMax-oMin))

		# Create an emtpy array to populate with the new image information
		imageM, imageN = self.imageSize()
		newImage = np.zeros([imageM, imageN],'uint8')

		for i in range(imageM):
			for j in range(imageN):
				newImage[i,j] = int((self.image[i,j] - oMin) * conversionRatio + nMin)
		self.updateImage(newImage, 'normaliseHistogram' + str(nMin) + '_' + str(nMax))

	def POequalise(self):

		# Create an empty array to populate with the new iamge information
		imageM, imageN = self.imageSize()
		newImage = np.zeros([imageM, imageN],'uint8')



	def PObitSlice(self, lMin, lMax):
		# Function to produce the parts of the image within a certain bit range, 
		# Returns a new image structure so that it can be used for generating masks
		newImageStruct = self.duplicate()
		newImage = newImageStruct.image

		lowerSubset = newImage < lMin
		midSubset = (newImage >= lMin) & (newImage <= lMax) # We want to include the values specified
		upperSubset = newImage > lMax

		newImage[lowerSubset] = 0
		newImage[midSubset] = 255
		newImage[upperSubset] = 0
			
		newImageStruct.updateImage(newImage,'GreySlice')
		return newImageStruct

	def POnegImage(self):
		# Function to invert the colours (or greyscale) of the image
		initialImage = self.image
		newImage = 255 - initialImage
		self.updateImage(newImage, 'Invert')






def __printSpacer__(*args):
	# Function created to print a line of asterix, made seperate to make code neater
	if (0 == len(args)): # If no arguments are included then print an asterix line spacer
		print('')
		print('************************************************************************************')
		print('')
	else:				# If arguments are provided then print the argument surrounded by asterix'
		print('')
		print('************************************************************************************')
		print(args[0])
		print('************************************************************************************')
		print('')



# a = Image('foetus.png')
# #b = Image()

# a.showImage()

# imshow('image', a.image)
# waitKey(0)
# destroyAllWindows()


#a = Image('NZjers1.png')
#b = Image('NZjers1.png')
a = Image('foetus.png')
#b = Image('foetus.png')

#makeHistogram(a)
#a.showImage()

def removeBlack(arg):
	mask = arg.PObitSlice(0,0)
	mask.POnegImage()
	arg.mask(mask)
	arg.showImage()



mask = a.PObitSlice(0,100)
#b.POnegImage()


#a.overlay(b,mask)
#combineImages(a,b,mask)
a.mask(mask)
a.showImage()
a.histogram()
a.POnormalise(0,255)
a.histogram()
a.showImage()


