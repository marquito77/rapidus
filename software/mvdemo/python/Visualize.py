import os, cv2
import numpy as np

# the following is copied from darknet framework to make bbox coloring consistent
# between darknet and mvdemo
baseColors = [ [1,0,1], [0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0] ]
colors = []

def _getColor(c, x, m):
    global baseColors
    ratio =  float(x)/float(m) * 5.
    i = int(np.floor(ratio));
    j = int(np.ceil(ratio));
    ratio -= float(i);
    r = (1.-ratio) * float(baseColors[i][c]) + ratio*float(baseColors[j][c]);

    return int(r*255.);


def initColors(numClasses):
    global colors
    colors = []
    for classId in range(numClasses):
        offset  = (classId*123457) % numClasses;
        red     = _getColor(2,offset,numClasses);
        green   = _getColor(1,offset,numClasses);
        blue    = _getColor(0,offset,numClasses);
        colors.append([blue, green, red])


def getColor(classId, numClasses):
    global colors
    if len(colors) != numClasses:
        initColors(numClasses)
        
    idx = classId % len(colors)

    return colors[idx]

def Visualize(img, results, numClasses):
    img_cp = img.copy()
    for r in results:
        clr     = getColor(r.objType, numClasses)
        txt     = r.name
        left    = r.left
        top     = r.top
        right   = r.right
        bottom  = r.bottom

        cv2.rectangle(img_cp, (left,top), (right,bottom), clr, thickness=3)
        cv2.rectangle(img_cp, (left,top-20),(right,top), clr,-1)
        cv2.putText(img_cp,txt,(left+5,top-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0), 2)

    return img_cp
