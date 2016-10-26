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
import cv2


# Modules below used for listing available images in the current directory
from os import getcwd
from os.path import join, split
from glob import glob

# Modules for function termination and writing to file
from sys import exit
import sys
import numpy as np

# Allow for plotting histogram graphs and 3d graphs http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Enable creating the windows breaking list connectivity
import copy
from math import*


# Multi-threading for faster operating when utilising windows
from threading import *

class Window:
	"""
	Attributes:
		x:		x coordinate of image centre point
		y:		y coordinate of image centrepoint
		width:	Width of the window
		content: NxN array containing the contents of the window

	"""
	def __init__(self,wholeImage, centerPoint, width):
		self.x, self.y = centerPoint
		self.content = copy.copy(wholeImage.image[ (self.x-width):(self.x+width+1), (self.y-width):(self.y+width+1) ])
		self.cp = width+1
		self.width = (2*width)+1

	def cpVal(self):
		# Returns the value of the pixel at the centre of the window
		return self.content[self.cp-1,self.cp-1]

	def average(self):
		## Averages the values within a window and then fills the window with the new average value
		averageValue = self.content.sum() / (self.width * self.width)

		self.content.fill(averageValue)

	def filter(self, filter):
		averageGaussianValue = (self.content * filter).sum() / filter.sum()
		self.content.fill(averageGaussianValue)

	def equalise(self, levels, ignore, offset):
		occurenceMap = np.zeros([256])
		greyScaleMap = np.zeros(256,'uint8')
		pixelCount=0
		for nVal in range(0,255):
			if nVal in ignore:
				occurenceMap[nVal] = 0
			else:
				occurenceMap[nVal] = (self.content==nVal).sum()
				pixelCount += occurenceMap[nVal]
		for n in range(256):
			## Generating incredibly low values for percentage chance (sigma function missing)
			greyScaleMap[n] =  ((np.sum(occurenceMap[0:n]) / pixelCount ) * (levels-1)) + offset
		self.content.fill(greyScaleMap[self.content[self.cp,self.cp]])

	def median(self):
		medianValue = np.median(self.content)
		self.content.fill(medianValue)

	def trimmedMean(self,nOmit):
		numValues = self.width * self.width
		halfPoint = ceil(numValues/2)
		values = np.reshape(self.content,numValues)



		order = values.argsort()
		ranks = order.argsort()

		print(str(ranks))





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

		### Should redo such that importing the image is a seperate step to constructing the shell

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

	def _genBorders(self, amount):
		bordered = self.image
		for val in range(amount):
			bordered = np.vstack([bordered , bordered[-1,:]])
			bordered = np.vstack([bordered[0,:] , bordered])
			bordered = bordered.transpose()
			bordered = np.vstack([bordered , bordered[-1,:]])
			bordered = np.vstack([bordered[0,:] , bordered])
			bordered = bordered.transpose()
		self.updateImage(bordered, 'Bordered')
		return bordered

	def _deBorders(self,amount):
		imageM, imageN = self.imageSize()
		debordered = self.image[amount:(imageM-amount), amount:(imageN-amount)]
		# [amount,(imageM - amount)], [amount,(imageN - amount)]
		self.updateImage(debordered, 'deBordered')


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
		newImage = np.zeros([imageM, imageN],'uint8')
		for i in range(imageM):
			for j in range(imageN):
				newImage[i,j] = self.image[i,j]
		newStruct.image = newImage
		
		newStruct.name = 'Duplicate' + self.name
		newStruct.transforms = self.transforms + ['Duplicate']
		return newStruct

	def mask(self, mask):
		# Function to apply a mask provided to the image
		self.image = np.minimum(self.image, mask.image)

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
		imwrite("./outputs/"+self.name + "_".join(self.transforms) + ".png", self.image)

	def histogram(self): 
		# Function to create a histogram of an image's greyscale levels using the formula:
		# p(r) = n / MN

		# Find the total number of elements in the image
		imageM, imageN = self.imageSize()
		imageMN = imageM * imageN *1.0
		print(imageMN)

		# Initialise an array for counting the occurences of each grey value
		nkCount = np.zeros(256)

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
		self.updateImage(newImage, 'normalise')

	def POequalise(self,levels,ignore,offset):
		""" 
		Inputs:
		self	= I dont understand object orientated enough to explain why self
		levels  = the number of unique luminence levels in the image
		ignore	= an array containing all luminence values to be ignored for the current array processing
		"""
		# Create an empty array to populate with the new iamge information
		imageM, imageN = self.imageSize()
		imageMN = imageM * imageN * 1.0 
		newImage = np.zeros([imageM, imageN],'uint8')

		# Initialise an array for counting the occurences of each grey value
		occurenceMap = np.zeros([256])
		greyScaleMap = np.zeros(256,'uint8')

		pixelCount = 0

		if ((levels + offset) > 255):
			__printSpacer__('Warning: DC-offset + desired level range exceeds 255, high values may show as black')
		
		for nVal in range(0,255):
			if nVal in ignore:
				occurenceMap[nVal] = 0
			else:
				occurenceMap[nVal] = (self.image==nVal).sum()
				pixelCount += occurenceMap[nVal]


		for n in range(256):
			## Generating incredibly low values for percentage chance (sigma function missing)
			greyScaleMap[n] =  ((np.sum(occurenceMap[0:n]) / pixelCount ) * (levels-1)) + offset


		for si in range(imageM):
			for sj in range(imageN):
				newImage[si,sj] = greyScaleMap[self.image[si,sj]]
		self.updateImage(newImage, 'equaliseHistogram' + '_' + str(levels) + '_' + str(ignore) + '_' + str(offset))

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

	def POfft(self):
		f = np.fft.fft2(self.image)
		fFhift = np.fft.fftshift(f)
		self.updateImage(fFhift,'fft')

	def POfftMag(self):
		# Function to perform a fast fourier transform on an image and subsequently
		# generate the resulting magnitude spectrum
		self.POfft()
		magnitude_spectrum = 20*np.log(np.abs(self.image))
		self.updateImage(magnitude_spectrum,'Magfft')

	def POifft(self):
		ifShift = np.fft.ifftshift(self.image)
		ifft = np.abs(np.fft.ifft2(ifShift))
		self.updateImage(ifft,'ifft')
		# Re normalise results to the range of grayscale values
		self.POnormalise(0,255)


		

	##########################################
	##										##
	## Methods (Group Operator Filters)		##
	##										##
	##########################################

	def createWindows(self, windowSize):
		## Function to create an array containing NxN sized window structures for the whole image
		imageM, imageN = self.imageSize()

		windowArray = []

		for i in range(windowSize,(imageM-windowSize)):
			for j in range(windowSize,(imageN-windowSize)):
				windowArray.append( Window(self,[i,j],windowSize))

		__printSpacer__(str(len(windowArray)) +' windows of size ' + str(windowSize*2 +1) +'x'+str(windowSize*2 +1)+ ' created from image')

		return windowArray
		
	def GOmean(self,windowSize):

		windowArray = self.createWindows(windowSize)

		imageM, imageN = self.imageSize()

		newImage = np.zeros([imageM, imageN],'uint8')

		for window in windowArray:
			window.average()
			newImage[window.x, window.y] = window.cpVal()

		self.updateImage(newImage, 'mean')

	def GOlinearGaussian(self, sigma):

		winSize = 2*(3*sigma) + 1

		self._genBorders(winSize+1)	

		imageM, imageN = self.imageSize()

		windowArray = self.createWindows(3*sigma)


		newImage = np.zeros([imageM, imageN],'uint8')

		# Make an empty array to contain the mask
		gausMask = np.zeros([winSize,winSize])

		# Populate the mask with calulated weightings
		for i in range(3*sigma+1):
			for j in range(3*sigma+1):
				gVal = exp( - ( pow(3*sigma - i,2) + pow(3*sigma - j,2) ) / ( 2 * sigma * sigma ) )

				# Gaussian mask is symettrical around centre so can set multiple values 
				# at the same time
				gausMask[i,j] = gVal
				gausMask[i,winSize-j-1] = gVal
				gausMask[winSize-i-1,j] = gVal
				gausMask[winSize-i-1,winSize-j-1] = gVal



		# Apply guasian mask
		for window in windowArray:
			window.filter(gausMask)
			newImage[window.x, window.y] = window.cpVal()

		self.updateImage(newImage, 'Gaussian')
		self._deBorders(winSize+1)

	def GOequalise(self, winSize, levels,ignore,offset):
		sys.exit() ## This code takes an age to run

		self._genBorders(winSize+1)	

		imageM, imageN = self.imageSize()
		
		windowArray = self.createWindows(winSize)

		newImage = np.zeros([imageM, imageN],'uint8')

		for window in windowArray:
			window.equalise(levels,ignore,offset)
			newImage[window.x, window.y] = window.cpVal()

		self.updateImage(newImage, 'equaliseWindowedHistogram' + '_' + str(levels) + '_' + str(ignore) + '_' + str(offset))

	##########################################
	##										##
	## Methods (Non Linear Filters)			##
	##										##
	##########################################

	def NLmedian(self, winSize):
		self._genBorders(winSize+1)	

		imageM, imageN = self.imageSize()

		windowArray = self.createWindows(winSize)
	
		newImage = np.zeros([imageM, imageN],'uint8')
	
		for window in windowArray:
			window.median()
			newImage[window.x, window.y] = window.cpVal()

		self.updateImage(newImage, 'Median Filter: r=' + str(winSize))
		self._deBorders(winSize+1)


	def NLmeanTrimmed(self, winSize,trim):
		self._genBorders(winSize+1)	

		imageM, imageN = self.imageSize()

		windowArray = self.createWindows(winSize)
	
		newImage = np.zeros([imageM, imageN],'uint8')

		if trim > (winSize*winSize):
			print('Please enter a trim value less than ' + str(winSize*winSize))
			return
	
		for window in windowArray:
			window.trimmedMean(trim)
			newImage[window.x, window.y] = window.cpVal()

		self.updateImage(newImage, 'Trimmed Mean Filter: r=' + str(winSize))
		self._deBorders(winSize+1)

	def NLmedianHuang(self,winSize):
		## Code found in https://nomis80.org/ctmf.pdf
		## Not original code
		self._genBorders(winSize+1)


		imageM, imageN = self.imageSize()

		newImage = np.zeros([imageM, imageN],'uint8')
		histogram = np.zeros([256])
		weightings = array(range(0,256))

		for i in range(2*winSize):
			for j in range(2*winSize):
				histogram[self.image[(i),(j)]] += 1

		for i in range(winSize,(imageM-winSize)):
			for j in range(winSize,(imageN-winSize)):
				for k in range(-winSize,winSize):
					histogram[self.image[(i+k),(j-winSize-1)]] -= 1
					histogram[self.image[(i+k),(j+winSize)]] += 1
				newImage[i,j] = np.median(histogram * weightings)

		self.updateImage(newImage,'HuangMedian')
		self._deBorders(winSize+1)









	##########################################
	##										##
	## Methods (Filter Applications)		##
	##										##
	##########################################

	def highlightRange(self,lBound,uBound,dcGain,levels):
		mask = self.PObitSlice(lBound,uBound)
		rangePixels = self.duplicate()
		rangePixels.mask(mask)
		rangePixels.POequalise(levels,[0],dcGain)
		mask.POnegImage()
		self.overlay(rangePixels,mask)

	def VisualiseFFT(self):
		self.POfftMag()
		self.POnormalise(0,255)









def __printSpacer__(*args):
	# Function created to print a line of asterix, made seperate to make code neater
	if (0 == len(args)): # If no arguments are included then print an asterix line spacer
		print('************************************************************************************')
		print('')
	else:				# If arguments are provided then print the argument surrounded by asterix'
		for i in range(len(args)):
			print('************************************************************************************')
			print('')
			print(args[i])
			print('')

	
	






# a = Image('foetus.png')
# #b = Image()

# a.showImage()

# imshow('image', a.image)
# waitKey(0)
# destroyAllWindows()


#a = Image('test.png')
a = Image('foetus.png')

a.NLmeanTrimmed(1, 0)

a.showImage()

#a = Image('NZjers1.png')
#b = Image('NZjers1.png')`
#a = Image('foetus.png')
#b = Image('foetus.png')



# for i in range(0,11):
# 	a = Image('foetus.png')
# 	a.NLmedian(i)
# 	a.saveImage()
# 	b = Image('NZjers1.png')
# 	b.NLmedian(i)
# 	b.saveImage()


