import json
import os
import sys
import shutil
import threading

def convertCocoYolo(jsonfile, classFilter=None, skipCrowd = False, balance = False):
            
    if not os.path.exists(jsonfile):
        print("Json-File does not exist {}".format(jsonfile))
        return
    
    jsonfile = os.path.abspath(jsonfile)
            
    fh = open(jsonfile, "r")
    if fh == -1:
        print("Could not open Json-File {}".format(jsonfile))
        return
    
    print("Reading json file...")
    jdb = json.load(fh)
    fh.close()
    print("    done!")
    
    imgs = jdb['images']
    annos = jdb['annotations']
    cats = jdb['categories']
    
    # remove gaps from category indices
    catsNames = [x['name'] for x in cats]
    if classFilter == None:
        classFilter = catsNames
        
    maxCatId = cats[-1]['id']
    catIdToYolo = [-1]*(maxCatId+1)
    yoloId = 0
    for x in classFilter:
        try:
            idx = catsNames.index(x)
        except ValueError:
            print("wrong class = {}: classes must contain one or more categories from {}".format(x, catsNames))
            return
        catId = cats[idx]['id']
        catIdToYolo[catId] = yoloId
        yoloId += 1
        
    personCatIdx = catsNames.index('person')
    personYoloId = catIdToYolo[cats[personCatIdx]['id']]
    print("Processing annotations...")
    annoMap = {}
    totalAnnosCreated = 0
    for i in range(len(annos)):        
        anno = annos[i]
        
        catId = anno['category_id']
        yoloId = catIdToYolo[catId]        
        if yoloId == -1:
            continue
        
        imgId = anno['image_id']
        bbox = anno['bbox']
        isCrowd = anno['iscrowd']
                                
        newElem = {'yoloId': yoloId, 'bbox': bbox, 'iscrowd': isCrowd}
        if imgId in annoMap:
            annoMap[imgId]['labels'].append(newElem)
        else:
            annoMap[imgId] =  {'labels': [newElem]}
            
        totalAnnosCreated += 1
        
    print("    done! Number of annotations created: {}".format(totalAnnosCreated))
    
    print("Adding image infos to annotations...")
    totalImages = 0
    crowdSkipCnt = 0
    classCnt = [0]*len(catsNames)
    for i in range(len(imgs)):
        img = imgs[i]

        imgId = img['id']
        if not imgId in annoMap:
            continue
        
        removeId = False
        if skipCrowd:
            for l in annoMap[imgId]['labels']:
                if l['iscrowd']:
                    removeId = True
                    crowdSkipCnt += 1
                    break
                
        if (not removeId) and (balance):
            labels = annoMap[imgId]['labels']
            numPers = 0
            numTotal = len(labels)
            for l in labels:
                classId = l['yoloId']
                if classId == personYoloId:
                    numPers += 1
            ratio = 0. if numTotal == 0 else float(numPers) / float(numTotal)
            if ratio > 0.66:
                removeId = True
                
        if removeId:
            del annoMap[imgId]
            continue
        
        for l in annoMap[imgId]['labels']:
            yoloId = l['yoloId']
            classCnt[yoloId] += 1
        
        annoMap[imgId]['file_name'] = img['file_name'];
        annoMap[imgId]['width']     = img['width'];
        annoMap[imgId]['height']    = img['height'];
        totalImages += 1
    print("    done! Number of images: {}".format(totalImages))
    if skipCrowd:
        print("    Removed {} images with 'crowd'".format(crowdSkipCnt))
    
    return {'annoMap': annoMap, 'classNames': classFilter, 'classCnt': classCnt}
    
def writeYoloAnno(anno, filename):
    labels = anno['labels']
    width = anno['width']
    height = anno['height']

    fh = None
    for label in labels:
        yoloId = label['yoloId']
        bbox = label['bbox']
        
        if yoloId < 0:
            continue
        
        if fh == None:
            fh = open(filename, "w")
        
        # create yolo label data
        x = bbox[0] + bbox[2]/2.
        x = x / width
        y = bbox[1] + bbox[3]/2.
        y = y / height
        w = bbox[2] / width
        h = bbox[3] / height

        fh.write(u"{} {:6f} {:6f} {:6f} {:6f}\n".format(yoloId, x, y, w, h))
    if fh != None:
        fh.close()
        
def writeFilelist(filelist, filename):
    with open(filename, "w") as fh:
        for f in filelist:
            f = f.replace(".txt", ".jpg")
            f = f.replace("\\", "/")
            fh.write(f + "\n")
            
def writeClassNames(classNames, filename):
    with open(filename, "w") as fh:
        for name in classNames:
            fh.write("{}\n".format(name))
            
def writeStats(classCnt, classNames, filename):
    with open(filename, "w") as fh:
        for i in range(len(classCnt)):
            cnt = classCnt[i]
            if cnt != 0:
                fh.write("{} {}\n".format(classNames[i], cnt))
                print("  {} {}".format(classNames[i], cnt))

def thrdCopyFiles(jpgFiles, targetDir, isOutputThread):
    n = len(jpgFiles)
    outputAt = int(n/100)
    cnt = 0
    cntPerc = 0
    for f in jpgFiles:
        shutil.copy2(f, targetDir)
        cnt += 1
        if (cnt >= outputAt):
            cntPerc += 1
            cnt = 0
            if isOutputThread and (cntPerc <= 100):
                #print(" {}%".format(cntPerc)),
                sys.stdout.write(" {}%".format(cntPerc))

def _createDb(cocoDir, targetDir, db, newDb, classFilter, balance, skipCrowd):
    cocoImgDir = os.path.join(cocoDir, db)
    if not os.path.isdir(cocoImgDir):
        print("Could not find coco image directory {}".format(cocoImgDir))
        return
    
    annoDir = os.path.join(cocoDir, "annotations")
    if not os.path.isdir(annoDir):
        print("Could not find annotation folder {}".format(annoDir))
        return

    annoFile = os.path.join(annoDir, "instances_{}.json".format(db))
    if not os.path.isfile(annoFile):
        print("Could not find annotation file {}".format(annoFile))
        return
    
    targetImgFolder = "{}_{}".format(db, newDb)
    targetImgDir = os.path.join(targetDir, targetImgFolder)
    maxTry = 1000
    cnt = 0
    while (cnt < maxTry) and (os.path.isdir(targetImgDir)):
        targetImgDir = os.path.join(targetDir, "{}_{:03}".format(targetImgFolder,cnt))
        cnt += 1
        
    # sanity check
    if os.path.isdir(targetImgDir):
        print("Internal error: target dir must not exist: {}".format(targetImgDir))
        return
    
    os.makedirs(targetImgDir)
    print("target dir will be {}".format(targetImgDir))
    
    data = convertCocoYolo(annoFile, classFilter, skipCrowd, balance)
    annos = data['annoMap']
    classCnt = data['classCnt']
    classNames = data['classNames']

    print("Writing yolo annotation files...")    
    txtPaths = []
    for imgId in annos:
        anno = annos[imgId]
        imgFilename = anno['file_name']
        txtFilename = imgFilename.replace('.jpg', '.txt')
        txtPath = os.path.join(targetImgDir, txtFilename)
        
        writeYoloAnno(anno, txtPath)
        txtPaths.append(txtPath)
    print("...done! Files written: {}".format(len(txtPaths)))        
    txtPaths.sort()
    
    print("Writing filelist, stats and class names...")
    filelistPath = os.path.join(targetImgDir, '_filelist.txt')
    statsPath    = os.path.join(targetImgDir, '_stats.txt')
    classesPath  = os.path.join(targetImgDir, '_classes.names')
    
    writeFilelist(txtPaths,     filelistPath)    
    writeStats(classCnt, classNames, statsPath)
    
    if classFilter == None:
        writeClassNames(classNames, classesPath)
    else:
        writeClassNames(classFilter, classesPath)
    print("...done!")
        
    print("Copying {} image files to target dir {}...".format(len(txtPaths), targetImgDir))
    jpgPaths = []
    for i in range(len(txtPaths)):
        f = txtPaths[i]
        jpgFile = os.path.basename(f)
        jpgFile = jpgFile.replace(".txt", ".jpg")
        jpgFile = os.path.join(cocoImgDir, jpgFile)
        if os.path.exists(jpgFile):
            jpgPaths.append(jpgFile)
        else:
            print("Could not find image file {}".format(jpgFile))
            continue
        
    numThreads = 4
    startIdx = 0
    endIdx = 0
    numPerThread = int(len(jpgPaths)/numThreads)
    thrds = []
    print("Start copying image files using {} threads...this might take a while...".format(numThreads))
    for i in range(numThreads):
        startIdx = endIdx
        endIdx += numPerThread
        if (i==numThreads-1):
            endIdx = len(jpgPaths)            
        workSet = jpgPaths[startIdx:endIdx]
        isOutputThread = (i==0)
        t = threading.Thread(target=thrdCopyFiles, args=(workSet,targetImgDir,isOutputThread,))
        t.start()
        thrds.append(t)
        
    for t in thrds:
        t.join()
    print("")
    print("Finished creating database in {}".format(targetImgDir))
    print("")
                

def createYoloDatabase(cocoDir, targetDir=None, classFilter=None, newDbName=None, 
                       balance=False, skipCrowd=True, valOnly=False):
    # check paths
    if not os.path.exists(cocoDir):
        print("Could not find dir {}".format(cocoDir))
        return        
    
    cocoDir = os.path.abspath(cocoDir)
        
    if targetDir == None:
        targetDir = cocoDir
    
    if not os.path.exists(targetDir):
        print("Creating targetDir {}".format(targetDir))
        os.makedirs(targetDir)
        
    if not os.path.exists(targetDir):
        print("Could not create targetDir {}".format(targetDir))
        return
    
    targetDir = os.path.abspath(targetDir)
    
    cocoDbs = ['val2017']
    if not valOnly:
        cocoDbs.append('train2017')
        
    for db in cocoDbs:
        print("processing database {}".format(db))
        _createDb(cocoDir, targetDir, db, newDbName, classFilter, balance, skipCrowd)
        