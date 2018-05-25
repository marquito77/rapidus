import cv2
import os
import numpy as np
import Visualize

def cfgGetVal(cfgfile, section, value):
    secStr = "[{}]".format(section)
    with open(cfgfile, "r") as fh:
        inSec = False
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                if line == secStr:
                    inSec = True
                else:
                    inSec = False
            if inSec and line.startswith(value):
                toks = line.split("=")
                if len(toks) != 2:
                    print("Error parsing cfgfile {}: Could not parse string {}".format(cfgfile, line))
                    return None
                vals = toks[1]
                vals = vals.strip()
                valToks = vals.split(",")
                ret = []
                for valStr in valToks:
                    try:
                        val = int(valStr)
                    except:
                        val = float(valStr)

                    ret.append(val)
                if len(ret) == 1:
                    ret = ret[0]
                print("cfgGetVal(): {}/{} = {}".format(section, value, ret))
                return ret
    print("Error: Could not find section/val ({}/{}) in file {}".format(section, value, cfgfile))
    return None

class BBox(object):
    def __init__(self, left, right, top, bottom, confidence, classId, className):
        self.left       = left
        self.right      = right
        self.top        = top
        self.bottom     = bottom
        self.confidence = confidence
        self.objType    = classId
        self.name       = className

def convertToBBoxes(results, offx, offy, xscale, yscale, imgw, imgh, classNames):
    boxes = []
    for res in results:
        l = int(res.left*imgw / xscale)-offx
        r = int(res.right*imgw / xscale)-offx
        t = int(res.top*imgh / yscale)-offy
        b = int(res.bottom*imgh / yscale)-offy
        c = res.confidence
        o = res.objType
        n = classNames[o]

        box = BBox(l, r, t, b, c, o, n)
        boxes.append(box)
    return boxes


def prepareImage(img, dim):
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

    #img01 = img.copy()/255.0
    #imgb[offy:offy+newh,offx:offx+neww,:] = resize(img01,(newh,neww),1)
    imgNet = cv2.resize(img, (neww, newh), cv2.INTER_LINEAR)
    imgNet = np.divide(imgNet, 255.)
    imgb[offy:offy+newh,offx:offx+neww,:] = imgNet
    im = imgb[:,:,(2,1,0)]
    im = im.astype(np.float16)

    ox = int(offx*imgw/neww)
    oy = int(offy*imgh/newh)
    sx = neww/dim[0]
    sy = newh/dim[1]
    return im, ox, oy, sx, sy

def prepareImage2(img, dim):
    im = cv2.resize(img, dim, cv2.INTER_LINEAR)
    im = np.divide(im, 255.)
    im = im[:,:,(2,1,0)]
    im = im.astype(np.float16)

    ox = 0
    oy = 0
    sx = 1.
    sy = 1.
    return im, ox, oy, sx, sy

def visualize(img, result, fps):
    img = Visualize.Visualize(img, result)
    img = cv2.putText(img, "%.2ffps" % fps, (70, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return img

def cleanQueue(queue):
    while not queue.empty():
        queue.get()
        queue.task_done()

class SingleImageProvider:
    def __init__(self, source):
        self.source = source
        self.img = None

    def nextImage(self):
        if not os.path.exists(self.source):
            print("Could not find file {}".format(self.source))
            return None

        if self.img is None:
            self.img = cv2.imread(self.source)

        return self.img

    def release(self):
        pass

class VideoImageProvider:
    def __init__(self, source):
        self.source = source
        self.cap = None

    def nextImage(self):
        if self.cap is None:
            if not os.path.isfile(self.source):
                print("Could not find file {}".format(self.source))
                return None
            self.cap = cv2.VideoCapture(self.source)

        if self.cap.isOpened():
            ret, img = self.cap.read()
            if ret == True:
                #img = prepareImage(img)
                return img
            else:
                print("Could not read from video caputre device = {}".format(self.cap))
                return None
        else:
            print("Video capture device is closed.")
            return None
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

