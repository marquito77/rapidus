import cv2
import time
from threading import Thread
#from multiprocessing import Process, Queue
from queue import Queue
from utils import *
from MvDetector import MvDetector

frameDrops = 0
imgQueue = Queue(10)
imgScaledQueue = Queue(10)
resQueue = Queue(10)
imgVisQueue = Queue(10)
threadDone = False
ox = 0
oy = 0
sx = 1.
sy = 1.
def thrdNextImage(ip, dim):
    global threadDone
    global ox, oy, sx, sy
    while not threadDone:
        #imgCpu = np.loadtxt("data_cpu.txt")        

        img = ip.nextImage()
        imgScaled, ox, oy, sx, sy = prepareImage(img, (dim,dim))

        #imgScaled = np.reshape(imgCpu, (3, 208, 208)).astype(np.float16)
        #imgScaled = np.transpose(imgScaled, (1,2,0))
        #imgScaled = imgScaled[:,:,(2,1,0)]

        imgQueue.put(img)
        imgScaledQueue.put(imgScaled)

def thrdMov(mvd, thresh):
    while not threadDone:
        imgScaled = imgScaledQueue.get()
        results = mvd.Detect(imgScaled, thresh)
        imgScaledQueue.task_done()
        resQueue.put(results)

# no performance gain
def thrdVisual():
    global threadDone
    while not threadDone:
        imgVis = imgVisQueue.get()
        cv2.imshow("Demo", imgVis)
        imgVisQueue.task_done()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            threadDone = True



imagefile = "/home/marquito/work/udacity/demo/perHanBac.jpg"
videofile = "/home/marquito/work/udacity/demo/mall.mp4"
graphfile = "/home/marquito/work/udacity/demo/models/puny-yolo1_208.graph"
#classNames = ["person", "bicycle", "suitcase"]
classNames = ["person"]
thresh = 0.25

cfgfile   = graphfile.replace(".graph", ".cfg")


dim = cfgGetVal(cfgfile, "net", "width")
ip  = VideoImageProvider(videofile)
#ip  = SingleImageProvider(imagefile)
mvd = MvDetector(graphfile)
det = mvd.GetDetector()

avrgFps = 10.
avrgTotalTime = 0.1
avrgAlpha = 0.05
numFrames = 0
startTime = time.time()
thrd = Thread(target=thrdNextImage, args=(ip, dim))
thrd.start()

thrdDet = Thread(target=thrdMov, args=(mvd,thresh))
#thrdDet.start()

thrdVis = Thread(target=thrdVisual)
#thrdVis.start()
while not threadDone:
    tTotal = time.time()

    #img = ip.nextImage()

    #t = time.time()
    #item = imgQueue.get()
    #img = item[0]
    #imgScaled = item[1]

    #if img is None:
     #   break

    #imgScaled, ox, oy, sx, sy = prepareImage(img, (DIM,DIM))
    imgScaled = imgScaledQueue.get()
    results = mvd.Detect(imgScaled, thresh)
    imgScaledQueue.task_done()
    #results = resQueue.get()
    img = imgQueue.get()

    t = time.time()
    results = det.Detect(results, thresh) # 4ms
    bboxes = convertToBBoxes(results, ox, oy, sx, sy, img.shape[1], img.shape[0], classNames)
    imgVis = visualize(img, bboxes, avrgFps)
    #print("time = {:.2f}ms".format(1000.*(time.time()-t)))

    #resQueue.task_done()
    #imgVisQueue.put(imgVis)
    imgQueue.task_done()

    cv2.imshow("Demo", imgVis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    numFrames += 1
    currTime = time.time()
    diffTime = currTime - startTime
    diffTotalTime = time.time() - tTotal
    avrgTotalTime += avrgAlpha * (diffTotalTime - avrgTotalTime)
    if (diffTime > 1.):
        avrgFps = numFrames / diffTime
        startTime = currTime
        numFrames = 0
        print("Avrg total time = {}ms".format(int(1000*avrgTotalTime)))
        print("fps = {:.1f}".format(avrgFps))
        if imgQueue.qsize() > 0:
            print("Queue size = {}".format(imgQueue.qsize()))


print("Wait for queues...")
cleanQueue(imgQueue)
cleanQueue(imgScaledQueue)
cleanQueue(resQueue)
cleanQueue(imgVisQueue)

print("Wait for threads...")
threadDone = True;
if thrd.isAlive():
    thrd.join()
if thrdDet.isAlive():
    thrdDet.join()
if thrdVis.isAlive():
    thrdVis.join()

print("Exiting...")
ip.release()
cv2.destroyAllWindows()
