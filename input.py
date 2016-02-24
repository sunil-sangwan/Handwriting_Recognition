""" script to take image as input and show it """
from skimage import io
import os
# image name will be replaced by pic01g
image=os.path.join('pic01.jpg')
#reading image from path
camra=io.imread(image)
from skimage.viewer import ImageViewer
viewer=ImageViewer(camra)
# print image
viewer.show()
#camra_mul=3*camra
