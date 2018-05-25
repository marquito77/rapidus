import rapidus as rpd

classes1 = ["person"]
classes10 = ["person", "bicycle", "stop sign", "backpack", "tie", 
           "cup", "banana", "orange", "laptop", "cell phone"]
rpd.createYoloDatabase("./coco", classFilter=classes1, newDbName="1class", valOnly=False)
rpd.createYoloDatabase("./coco", classFilter=classes10, newDbName="10class", valOnly=False)