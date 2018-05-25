import os
import matplotlib.pyplot as plt
import numpy as np

def movingAvrg(data, alpha=0.002):
    a = [0]*len(data)
    a[0] = data[0]
    for i in range(1,len(data)):
        d = data[i]
        a[i] = (1.-alpha)*a[i-1] + alpha*d
    return a
        

def drawLossFromLog(logfile, metrics = None, maxIter = None, logscale=False,
                    alpha=None):
    if not os.path.exists(logfile):
        print("Could not find file {}".format(logfile))
        return
    
    if metrics == None:
        metrics = ["loss"]
    
    avrgLoss = []
    avrgObjProps = []
    avrgNoObjProps = []
    avrgIouProps = []
    avrgClassProps = []
    avrgRecallProps = []
    with open(logfile, "r") as fh:        
        objProps = []
        noObjProps = []
        iouProps = []
        classProps = []
        recallProps = []
        for lineStr in fh:
            lineStr = lineStr.strip()
            if lineStr.startswith("Region"):
                toks = lineStr.split(', ')
                if len(toks) != 6:
                    break
                
                iouStr = toks[0]
                iouVal = iouStr.split(':')
                iouVal = iouVal[1].strip()

                classStr = toks[1]
                classVal = classStr.split(':')
                classVal = classVal[1].strip()                
     
                objStr = toks[2]
                objVal = objStr.split(':')
                objVal = objVal[1].strip()                
                                
                noObjStr = toks[3]
                noObjVal = noObjStr.split(':')
                noObjVal = noObjVal[1].strip()                

                recallStr = toks[4]
                recallVal = recallStr.split(':')
                recallVal = recallVal[1].strip()                

                try:
                    iouVal = float(iouVal)
                    classVal = float(classVal)
                    objVal = float(objVal)
                    noObjVal = float(objVal)
                    recallVal = float(recallVal)
                except:
                    continue
 
                iouProps.append(iouVal)
                classProps.append(classVal)
                objProps.append(objVal)
                noObjProps.append(noObjVal)
                recallProps.append(recallVal)
     
                
            if lineStr.endswith("images"):
                avrgObjProps.append(np.mean(objProps))
                avrgNoObjProps.append(np.mean(noObjProps))
                avrgIouProps.append(np.mean(iouProps))
                avrgClassProps.append(np.mean(classProps))
                avrgRecallProps.append(np.mean(recallProps))
                
                objProps = []
                noObjProps = []
                iouProps = []
                classProps = []
                recallProps = []
                
                toks = lineStr.split(",")
                if len(toks) < 2:
                    print("Unknown log format: Could not split line by \",\": {}".format(lineStr))
                    return
                avrgTok = toks[0]
                avrgTok = avrgTok.strip()
                avrgValToks = avrgTok.split(": ")
                if len(avrgValToks) < 2:
                    print("Unknown log format")
                    return                
                avrgLoss.append(float(avrgValToks[1]))
            if maxIter != None:
                if maxIter <= len(avrgLoss):
                    break
    
    if alpha != None:
        avrgLoss        = movingAvrg(avrgLoss,          alpha)
        avrgObjProps    = movingAvrg(avrgObjProps,      alpha)
        avrgNoObjProps  = movingAvrg(avrgNoObjProps,    alpha)
        avrgIouProps    = movingAvrg(avrgIouProps,      alpha)
        avrgClassProps  = movingAvrg(avrgClassProps,    alpha)
        avrgRecallProps = movingAvrg(avrgRecallProps,   alpha)
    
    if (logscale):
        avrgObjProps    = -np.log(avrgObjProps)
        avrgNoObjProps  = -np.log(avrgNoObjProps)
        avrgIouProps    = -np.log(avrgIouProps)
        avrgClassProps  = -np.log(avrgClassProps)
        avrgRecallProps = -np.log(avrgRecallProps)
            
    fig,ax = plt.subplots()
    for m in metrics:
        if m == "loss":
            data = avrgLoss
        elif m == "obj":
            data = avrgObjProps
        elif m == "noobj":
            data = avrgNoObjProps
        elif m == "iou":
            data = avrgIouProps
        elif m == "class":
            data = avrgClassProps
        elif m == "recall":
            data = avrgRecallProps
        else:
            print("Unknown metric {}".format(m))
            return
        
        x = range(len(data))
        ax.plot(x, data, label=m)
    ax.set_xlim([0, len(data)*1.3])
    ax.set_ylim([0, 5])
    ax.set_xlabel("iterations")
    ax.set_ylabel("loss")
    ax.grid()
    ax.legend()
    plt.show()
    
    return ax
        