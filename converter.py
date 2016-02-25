import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
from skimage.viewer import ImageViewer
""" for taking image as input """
from skimage import io
import os
#image name replace hare by pico1.jpg
file_path=os.path.join('pic01.jpg')

# read image
image=io.imread(file_path)

# load classifier
clf=joblib.load("digits_cls.pkl")

""" convert image from rgb to gray_scale"""
from skimage.color import rgb2gray
gray_image=rgb2gray(image)

""" apply gaussian filter """
try:
	from skimage import filters
except ImportError:
	from skimage import filter as filters
gaussian_img=filters.gaussian_filter(gray_image,(5,5),0)

""" set threshold_otsu for convert image in binary image image """
# it optimaly set threshold value
from skimage.filters import threshold_otsu
thresh=threshold_otsu(gaussian_img)
binary=gaussian_img>thresh

""" find contours in image"""
from skimage import measure
new_b=binary
contours=measure.find_contours(binary)

# make rectangle for digits
from skimage.morphology import rectangle
rectangles=[rectangle(contour) for contour in contours]
