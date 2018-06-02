DATA_FILE=./data/training/rapidus-hagl10.data
MODEL_FILE=./data/models/rapidus-hagl10.cfg
WEIGHTS_FILE=./data/models/rapidus-hagl10.weights

darknet/darknet detector test ${DATA_FILE} ${MODEL_FILE} ${WEIGHTS_FILE} data/media/perBikeStop.jpg