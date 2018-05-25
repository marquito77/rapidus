from libpydetector import YoloDetector
import os, io, time
import numpy as np
from mvnc import mvncapi as mvnc
from skimage.transform import resize
import cv2
from utils import cfgGetVal

#NUM_CLASSES = 1
#THRESH = 0.6
#IMG_WD = 416
BLOCK_WD = 13
TARGET_BLOCK_WD = 13
NMS = 0.4

def createYoloDetector(numClasses, anchors):
    # parse cfg file to find dimensions, #classes etc.
    #classes = cfgGetVal(cfgfile, "region", "classes")
    #biases = cfgGetVal(cfgfile, "region", "anchors")
    c = (numClasses+5) * 5

    return YoloDetector(1, c, BLOCK_WD, BLOCK_WD, numClasses, 
                        NMS, TARGET_BLOCK_WD, anchors)

class MvDetector():
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
    devices = mvnc.EnumerateDevices()
    devNum = len(devices)
    if len(devices) == 0:
        print('No MVNC devices found')
        quit()
    devHandle = []
    graphHandle = []
    def __init__(self, graphfile, cfgfile = None):       
        if cfgfile == None:
            cfgfile = graphfile.replace(".graph", ".cfg")

        if not os.path.exists(graphfile):
            print("Error: Could not find graph file {}".format(graphfile))
            return
        if not os.path.exists(cfgfile):
            print("Error: Could not find darknet config file {}".format(cfgfile))
            return

        for i in range(MvDetector.devNum):
            MvDetector.devHandle.append(mvnc.Device(MvDetector.devices[i]))
            MvDetector.devHandle[i].OpenDevice()
            opt = MvDetector.devHandle[i].GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
            # load blob
            with open(graphfile, mode='rb') as f:
                blob = f.read()
            MvDetector.graphHandle.append(MvDetector.devHandle[i].AllocateGraph(blob))
            MvDetector.graphHandle[i].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
            #MvDetector.graphHandle[i].SetGraphOption(mvnc.GraphOption.DONTBLOCK, 1)
            iterations = MvDetector.graphHandle[i].GetGraphOption(mvnc.GraphOption.ITERATIONS)

        imgWidth   = cfgGetVal(cfgfile, "net", "width")
        imgHeight  = cfgGetVal(cfgfile, "net", "height")
        numClasses = cfgGetVal(cfgfile, "region", "classes")
        anchors    = cfgGetVal(cfgfile, "region", "anchors")

        self.detector = createYoloDetector(numClasses, anchors)

        self.dim = (imgWidth, imgHeight)
        self.blockwd = BLOCK_WD
        self.wh = BLOCK_WD*BLOCK_WD
        self.targetBlockwd = TARGET_BLOCK_WD
        self.classes = numClasses
        self.nms = NMS


    def __del__(self):
        for i in range(MvDetector.devNum):
            MvDetector.graphHandle[i].DeallocateGraph()
            MvDetector.devHandle[i].CloseDevice()
    def PrepareImage(self, img, dim):
        tPrep0 = time.time()

        imgw = img.shape[1]
        imgh = img.shape[0]
        imgb = np.empty((dim[0], dim[1], 3))
        imgb.fill(0.5)

        if imgh/imgw > dim[1]/dim[0]:
            neww = int(imgw * dim[1] / imgh)
            newh = dim[1]
        else:
            newh = int(imgh * dim[0] / imgw)
            neww = dim[0]
        offx = int((dim[0] - neww)/2)
        offy = int((dim[1] - newh)/2)

        tDiff0 = 1000.*(time.time() - tPrep0)
        print("      prep0 = {:.2f}".format(tDiff0))

        tRes = time.time()

        #img01 = img.copy()/255.0
        #imgb[offy:offy+newh,offx:offx+neww,:] = resize(img01,(newh,neww),1)
        imgNet = cv2.resize(img, (neww, newh), cv2.INTER_LINEAR)
        imgNet = np.divide(imgNet, 255.)
        imgb[offy:offy+newh,offx:offx+neww,:] = imgNet
        im = imgb[:,:,(2,1,0)]

        tResDiff = 1000.*(time.time() - tRes)
        print("      resize = {:.2f}".format(tResDiff))

        return im, int(offx*imgw/neww), int(offy*imgh/newh), neww/dim[0], newh/dim[1]

    def _reshape(self, out):
        shape = out.shape
        out = np.transpose(out.reshape(self.wh, int(shape[0]/self.wh)))  
        out = out.reshape(shape)
        return out

    def GetDetector(self):
        return self.detector

    def Detect(self, img, thresh):
        #imgw = img.shape[1]
        #imgh = img.shape[0]

        #im,offx,offy,xscale,yscale = self.PrepareImage(img, self.dim)
        #print('xscale = {}, yscale = {}'.format(xscale, yscale))
        #print("in shape = {}".format(img.shape))
        MvDetector.graphHandle[0].LoadTensor(img, 'user object')
        out, userobj = MvDetector.graphHandle[0].GetResult()
        #print("out shape = {}".format(out.shape))
        out = self._reshape(out)
        out = out.astype(np.float32)

        #with open("mvout.txt", "w") as fh:
        #    outStr = [str(x) for x in out]
        #    fh.writelines("\n".join(outStr))


        #print("out shape after reshape = {}".format(out.shape))

        #out = self.detector.Detect(out, thresh) # 4ms
        #pyresults = [BBox(x,xscale,yscale, offx, offy) for x in internalresults]

        return out
