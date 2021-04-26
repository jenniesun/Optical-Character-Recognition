# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import tensorflow as tf
import string

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

##############################Text Localization#################################

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(orig_H, orig_W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = orig_W / float(newW)
rH = orig_H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < args["min_confidence"]:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
# define an empty to store matrices that represent the bounding boxes
cropped = {}
# define a padding value
padding = 0
starts = {}
ends = {}
# loop over the bounding boxes
for index, (startX, startY, endX, endY) in enumerate(boxes):
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
	# add padding to the bounding box
	if startY - padding >= 0:
		startY -= padding
	if endY + padding <= orig_H:
		endY += padding
	if startX - padding >= 0:
		startX -= padding
	if endX + padding <= orig_W:
		endX += padding
	# store the cropped image
	cropped[index] = orig[startY:endY, startX:endX]
	starts[index] = (startX, startY)
	ends[index] = (endX, endY)
# 	# draw the bounding box on the image
# 	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
#
# # show the output image
# cv2.imshow("Text Detection", orig)
# cv2.waitKey()

##############################Text Segmentation#################################
def text_preprocess(text,
					index,
					blur_size = (3,3),
					binary_thres = (130,255),
					morph_size = (3,3),
					):
	# convert to grayscale and blur the image
	gray = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,blur_size,0)
	# Applied inversed thresh_binary
	binary = cv2.threshold(blur, binary_thres[0], binary_thres[1],
						   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	# perform line thining
	kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_size)
	thre_mor = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3)

	# cv2.imshow('binary', binary)
	# cv2.waitKey(0)
	cv2.imwrite('./preprocessing/blur-{}.png'.format(index), blur)
	cv2.imwrite('./preprocessing/binary-{}.png'.format(index), binary)
	cv2.imwrite('./preprocessing/dilution-{}.png'.format(index), thre_mor)

	return binary, thre_mor


def char_segmentation(texts, pad = 5, resize = (28,28)):
	# define a dict for storing segmented characters
	text_chars = {}
	# check if the input contains text
	if (texts):
		# loop through the input dictionary
		for index, text in texts.items():
			# preprocess the text
			binary, thre_mor = text_preprocess(text, index)
			#find contours
			ctrs, _ = cv2.findContours(thre_mor,
									   cv2.RETR_EXTERNAL,
									   cv2.CHAIN_APPROX_SIMPLE)
			# sort contours
			sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
			# define an empty list for storing recognized character
			text_chars[index] = []
			# loop through each contour
			for i, ctr in enumerate(sorted_ctrs):
				# Get bounding box
				(x, y, w, h) = cv2.boundingRect(ctr)
				# ignore anything whose height is less than half of the
				# original text
				if h < (text.shape[0]/2):
					continue
				# Getting ROI based on the binary image
				roi = thre_mor[y:y + h, x:x + w]
				# perform zero padding
				roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad,
										 cv2.BORDER_CONSTANT)
				# invert image
				roi = 255 - roi
				# resize to 28 x 28
				roi = cv2.resize(roi, resize)
				# save the matrix
				text_chars[index].append(roi)
				# write image
				cv2.imwrite('./characters/roi_imgs {}-{}.png'.format(index, i), roi)
				pass
			pass
		pass

	return(text_chars)


def make_prediction(chars_dict, model_path):
	# create a mapping
	nums = list(string.digits)
	upper = list(string.ascii_uppercase)
	other = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
	classes = nums + upper + other

	mapping = {}
	# loop through classes to fill the mapping dictionary
	for i, value in enumerate(classes):
		mapping[i] = value
	# load model
	my_model = tf.keras.models.load_model(model_path)
	# create an empty dictionary for storing prediction results
	my_prediction = {}
	# loop through input dictionary of segmented texts
	for index, chars in chars_dict.items():
		# create an empty string for storing predicted characters
		my_prediction[index] = ''
		# loop through characters
		for char in chars:
			# normalize
			char = char/255
			# change dimension to work with neural network model input
			char = np.expand_dims(char, axis = -1)
			char = np.expand_dims(char, axis = 0)
			# make prediction
			pred = np.argmax(my_model.predict(char))
			# identify corresponding character
			pred = mapping[pred]
			# add to stored characters
			my_prediction[index] += pred
			pass
		pass

	return my_prediction

my_charactors = char_segmentation(cropped, pad = 5)

my_prediction = make_prediction(my_charactors, './LeNet_EMNIST&Char74k_trained')

print(my_prediction)

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 2
font_thickness = 4
# cv2.imshow("Original", orig)
my_pic = cv2.rectangle(orig, starts[0], ends[0], (0, 0, 255), 2)
# my_pic = cv2.rectangle(my_pic, starts[1], ends[1], (0, 0, 255), 2)
my_pic = cv2.putText(my_pic, my_prediction[0], (starts[0][0], starts[0][1] - 10),
					 font, font_size, (0, 0, 255), font_thickness, cv2.LINE_AA)
# my_pic = cv2.putText(my_pic, my_prediction[1], (starts[1][0], starts[1][1] - 5),
# 					 font, font_size, (0, 0, 255), font_thickness, cv2.LINE_AA)
cv2.imwrite('text.jpg',my_pic)
