# USAGE
# python indexer.py --dataset images --index index.cpickle

# import the necessary packages
from feature_extractor import RGBHistogram
import argparse
import cPickle
import pickle
import glob
import cv2
import os

# constructing the argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required =True,
				help = "Path to directory that contains images to be indexed")
ap.add_argument("-i", "--index", required=True,
				help="Path to where computed index will be stored")
args=vars(ap.parse_args())

# initializing the index dictionary to store feature vectors of images
# key = the name of image
# value = feature vector of image (size = (8*8*8,)   )
index={}

# initialize image descriptor - 3D histogram with 8 bits per channel
desc=RGBHistogram([8,8,8])

# using glob to grab the image paths and loop over them
for image_path in glob.glob(args["dataset"]+"/*.jpg"):
	# extract the key for index id which is actually the file name
	# k=image_path.split(os.sep)[-1]
	k = image_path[image_path.rfind("/") + 1:]

	# loading the image and getting its feature vector through RGBHistogram.describe()
	image=cv2.imread(image_path)
	features= desc.describe(image)
	index[k] = features

# writing index on to disc
output=open("index.pkl",'wb')
pickle.dump(index, output)
output.close()


# show how many images we indexed
print "done...indexed %d images" % (len(index))

