import cv2
import time
from threading import Thread
import argparse
import os
#from multiprocessing import Process, Queue
from queue import Queue
from utils import *
from MvDetector import MvDetector

frameDrops = 0
imgQueue = Queue(10)
imgScaledQueue = Queue(10)
resQueue = Queue(10)
imgVisQueue = Queue(10)
stopThreads = False

def thrdNextImage(vc, dim):
    global stopThreads
    while not stopThreads:
        ret, img = vc.read()
        if not ret:
            print("No images left")
            imgQueue.put(None)
            imgScaledQueue.put([None, None])
            #stopThreads = True
            break
        imgScaled, scalingData = prepareImage(img, (dim,dim))
        imgQueue.put(img)
        imgScaledQueue.put([imgScaled, scalingData])
    print("thrdNextImage: exit thread")

def thrdMov(mvd, thresh):
    global stopThreads
    while not stopThreads:
        [imgScaled, scalingData] = imgScaledQueue.get()
        if imgScaled is None:
            imgScaledQueue.task_done()
            break
        results = mvd.Detect(imgScaled, thresh)
        imgScaledQueue.task_done()
        resQueue.put([results, scalingData])
    print("thrdMov: exit thread")

def thrdWrite(filename):
    global stopThreads
    vw = None
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    print("fourcc".format(fourcc))
    fps = 20
    while not stopThreads:
        img = imgVisQueue.get()
        if img is None:
            imgVisQueue.task_done()
            break

        if vw is None:
            vw = cv2.VideoWriter(filename, fourcc, fps, (img.shape[1], img.shape[0]), True)
            if not vw.isOpened():
                print("Could not write video to {}".format(filename))
                break
            else:
                print("Saving video to {}".format(filename))
        vw.write(img)
        imgVisQueue.task_done()
    if not vw is None:
        vw.release()
    print("thrdWrite: exit thread")


def mvdemo(source, cfgFile, thresh, outFile=""):
    global stopThreads

    if not os.path.isfile(source):
        print("Could not find source {}".format(source))
        return

    if not os.path.isfile(cfgFile):
        print("Could not find model cfg {}".format(cfgFile))
        return

    graphFile = cfgFile.replace(".cfg", ".graph")
    if not os.path.isfile(graphFile):
        print("Could not find graph file {}".format(graphFile))
        print("Graph file must be in same dir as cfg file.")
        return

    namesFile = cfgFile.replace(".cfg", ".names")
    if not os.path.isfile(namesFile):
        print("Could not find names file {}".format(namesFile))
        print("Names file must be in same dir as cfg file.")
        return

    imgExts = (".jpg", ".dib", ".jpeg", ".jpg", ".jpe", 
               ".png", ".pbm", ".pgm", ".ppm", ".tiff", ".tif")
    isImage = True if source.lower().endswith(imgExts) else False

    # timeout = 0: show image once
    # timeout = 1: show images in loop (for video and cam)
    waitforTimeout = 0 if isImage else 1

    if outFile == "":
        writeOutput = False 
    else: 
        writeOutput = True
        outFile = os.path.abspath(outFile)


    classNames = []
    with open(namesFile, "r") as f:
        for line in f:
            line = line.strip()
            classNames.append(line)

    inputWidth = cfgGetVal(cfgFile, "net", "width")
    numClasses = cfgGetVal(cfgFile, "region", "classes")
    vc = cv2.VideoCapture(source)
    if not vc.isOpened():
        print("Could not open source {}".format(source))
        return

    mvd = MvDetector(graphFile)
    det = mvd.GetDetector()

    thrdImg = Thread(target=thrdNextImage, args=(vc, inputWidth))
    thrdDet = Thread(target=thrdMov, args=(mvd,thresh))
    #thrdVis = Thread(target=thrdVisual)
    thrdWrt = Thread(target=thrdWrite, args=(outFile,))

    thrdImg.start()
    thrdDet.start()
    #thrdVis.start()

    if writeOutput and not isImage:
        thrdWrt.start()

    fps = 20.
    numFrames = 0
    startTime = time.time()
    # our main thread handles conversion and visualization
    while not stopThreads:
        [results, scalingData] = resQueue.get()
        if results is None:
            break

        bboxesRaw = det.Detect(results, thresh) # process final region layer
        resQueue.task_done()

        # convert and visualize boxes
        bboxes = convertToBBoxes(bboxesRaw, scalingData, classNames)
        imgOrig = imgQueue.get()
        if imgOrig is None:
            break
        imgVis = visualize(imgOrig, bboxes, fps, numClasses)
        imgQueue.task_done()

        if writeOutput:
            if isImage:
                cv2.imwrite(outFile, imgVis)
                print("Saved image to {}".format(outFile))
            else:
                imgVisQueue.put(imgVis)

        # render imgVis
        cv2.imshow("Demo", imgVis)
        # stop program with key 'q'
        if (cv2.waitKey(waitforTimeout) & 0xFF) in (ord('q'), 27):            
            break

        # update fps info
        numFrames += 1
        currTime = time.time()
        tDiff = currTime - startTime
        if tDiff > 1.:
            fps = numFrames / tDiff
            startTime = currTime
            numFrames = 0           

    stopThreads = True

    # put None into each queue to wake up threads which are waiting for queue.get()
    print("Cleaning up queues")
    cleanQueue(imgQueue)
    cleanQueue(resQueue)
    cleanQueue(imgVisQueue)

    print("Waking up threads")
    imgQueue.put(None, timeout=1)
    resQueue.put(None, timeout=1)
    imgVisQueue.put(None, timeout=1)

    print("Wait for threads to stop")
    if thrdImg.isAlive():
        thrdImg.join()
    if thrdDet.isAlive():
        thrdDet.join()
    if thrdWrt.isAlive():
        thrdWrt.join()

    vc.release()
    cv2.destroyAllWindows()

    print("Exit demo")

def main():
    parser = argparse.ArgumentParser(description='Run a darknet model on the NCS',
                                     epilog="Graph and names file must lie next to cfg file")
    parser.add_argument("source",         type=str,   help="a video/image file or a webcam device")
    parser.add_argument("cfg",            type=str,   help="a darknet model .cfg file")
    parser.add_argument("-t", "--thresh", type=float, help="threshold for object detection", default=0.25)
    parser.add_argument("-o", "--output", type=str,   help="output file for video or image", default="")

    args = parser.parse_args()
    mvdemo(args.source, args.cfg, args.thresh, args.output)

if __name__ == "__main__":
    main()