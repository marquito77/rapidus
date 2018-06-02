import rapidus as rpd

rpd.convertYoloToCaffe("./data/models/rapidus-1.cfg", "./data/models/rapidus-1.weights")
print()
rpd.convertYoloToCaffe("./data/models/rapidus-hagl10.cfg", "./data/models/rapidus-hagl10.weights")