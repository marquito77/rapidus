def downloadCoco(targetDir, valOnly=False):
    annosUrl = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    valUrl   = "http://images.cocodataset.org/zips/val2017.zip"
    trainUrl = "http://images.cocodataset.org/zips/train2017.zip"
    
    try:
        import wget
    except:
        print("Missing module 'wget': Please run")
        print("    pip install wget")
        return

    import os.path
    import zipfile
    import tempfile
    
    if not os.path.isdir(targetDir):
        try:
            os.makedirs(targetDir)        
        except:
            print("Could not find and create target dir {}".format(targetDir))
            return
        
    targetDir = os.path.abspath(targetDir)
    tempDir = tempfile.gettempdir()
    
    if valOnly:
        urls = [annosUrl, valUrl]
    else:
        urls = [annosUrl, valUrl, trainUrl]
    fnames = []
    
    print("Start downloading COCO data. This might take a long time!")
    for u in urls:
        print("downloading {}".format(u))
        fname = wget.download(u, tempDir)
        fnames.append(fname)
    print("Finished downloading COCO data")
    
    for f in fnames:
        print("Unzipping {} to dir {}".format(f, targetDir))
        zipRef = zipfile.ZipFile(f, 'r')
        zipRef.extractall(targetDir)
        zipRef.close()
        
    print("Cleaning up zip files")
    for f in fnames:
        os.remove(f)