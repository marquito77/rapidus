import caffe
from PIL import Image
import numpy as np
import skimage
import skimage.io
import skimage.transform

def dumpData(blobs, key):
	fname = "{}.txt".format(key)
	data = blobs[key].data
	data = data.flatten()
	np.savetxt(fname, data, fmt="%.6f")



imagepath = "/home/marquito/work/udacity/demo/perHanBac.jpg"
protopath = "/home/marquito/work/YoloV2NCS-master/models/caffemodels/puny-yolo1_208.prototxt"
modelpath = protopath.replace(".prototxt", ".caffemodel")

net = caffe.Net(protopath, 1, weights=modelpath)
inputShape = net.blobs["data"].data.shape

img = np.loadtxt("data_cpu.txt")
data = np.reshape(img, (1, 3, 208, 208))

#img = skimage.io.imread(imagepath)
#data = skimage.img_as_float(img).astype(np.float32)
#data = skimage.transform.resize(data, inputShape[2:])
#data = np.transpose(data, (2, 0, 1))
#data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))

net.blobs['data'].data[...] = data
out = net.forward()

print("img shape = {}".format(img.shape))
print("input data shape = {}".format(data.shape))
print("input shape = {}".format(net.blobs["data"].data.shape))
print("out shape = {}".format(out["conv9"].shape))
print("net blobs keys = {}".format(net.blobs.keys()))

keys = net.blobs.keys()
for key in keys:
	dumpData(net.blobs, key)