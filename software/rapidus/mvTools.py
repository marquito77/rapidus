from mvnc import mvncapi as mvnc
from .utils import cfgGetVal
import os.path

# todo: get this value from .graph file (which is possible)
BLOCK_WD = 13

class MvDetector():
    def __init__(self, graphfile, cfgfile = None):       
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        self.devices = mvnc.EnumerateDevices()
        self.devNum = len(self.devices)
        if len(self.devices) == 0:
            print('No MVNC devices found')
            quit()
        self.devHandle = []
        self.graphHandle = []

        if cfgfile == None:
            cfgfile = graphfile.replace(".graph", ".cfg")

        if not os.path.exists(graphfile):
            print("Error: Could not find graph file {}".format(graphfile))
            return
        if not os.path.exists(cfgfile):
            print("Error: Could not find darknet config file {}".format(cfgfile))
            return

        for i in range(self.devNum):
            self.devHandle.append(mvnc.Device(self.devices[i]))
            self.devHandle[i].OpenDevice()

            # load blob
            with open(graphfile, mode='rb') as f:
                blob = f.read()
            self.graphHandle.append(self.devHandle[i].AllocateGraph(blob))
            self.graphHandle[i].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
            #MvDetector.graphHandle[i].SetGraphOption(mvnc.GraphOption.DONTBLOCK, 1)

        imgWidth   = cfgGetVal(cfgfile, "net", "width")
        imgHeight  = cfgGetVal(cfgfile, "net", "height")
        numClasses = cfgGetVal(cfgfile, "region", "classes")
        anchors    = cfgGetVal(cfgfile, "region", "anchors")

        #self.detector = createYoloDetector(numClasses, anchors)

        self.dim = (imgWidth, imgHeight)
        self.blockwd = BLOCK_WD
        self.wh = BLOCK_WD*BLOCK_WD
        self.targetBlockwd = BLOCK_WD
        self.classes = numClasses
        self.nms = 0.4

    def __del__(self):
        for i in range(self.devNum):
            self.graphHandle[i].DeallocateGraph()
            self.devHandle[i].CloseDevice()

    def Detect(self, img):
        self.graphHandle[0].LoadTensor(img, 'user object')
        out, userobj = self.graphHandle[0].GetResult()

        return out