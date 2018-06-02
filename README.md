# rapidus
Embedded Real-time Object Detection and Classification using the Raspberry Pi and Movidius Neural Compute Stick

## Introduction
This is the repository for my Udacity Machine Learning Capstone project. The main goal of my project was to develop object detection CNN models which are small enough to run in 50ms on a Movidius NCS. There are two models in the repository: a one-class model to detect people and a multiclass model which knows 10 classes.

If you don't own a NCS you can still train and test the models as I did but you won't be able to run the NCS demo tool.

## Reports
- [Final Report](reports/report.pdf)
- [Proposal](reports/proposal.pdf)

## Requirements
- Linux with Ubuntu 16.04 (possibly on a virtual machine)
- Raspberry Pi running Stretch
- Movidius NCS connected to host or RPi
- Full installation of Movidius SDK (NCAPI v1) with OpenCV

## Demo (using NCS)
There is a python and a C++ version of the demo application. Both do the same: they process a short video file running the people detector model and showing results via bounding boxes.

Connect the NCS and run one of the demo application.

To run the python demo:
```
sh runDemoPython.sh
```

To run the C++ demo:
```
cd mvdemo/cpp
make
cd ../..
sh runDemoCpp.sh
```

This should work on a PC/Linux as well as on the Raspberry Pi.

## Demo (using darknet, not NCS)
To test my 10-class model on one image:
```
sh runDarknetImageDemo.sh
```

## Training
If you want to reproduce the training as described in my report you need to do the following from root directory:
1. download and build the darknet fork of Alexey 
`sh runDownloadDarknet`
2. download COCO
`python runDownloadCoco.py`
3. create COCO datasets
`python runCreateDatabase.py`
4. start training
`darknet/darknet detector train data/training/rapidus-1.data data/models/rapidus-1.cfg`
Training should run very slow because GPU and CUDA support in darknet is turned off by default. Edit darknet/Makefile and turn on 
GPU=1, CUDA=1 and OPENCV=1

## Validation
To calculate the mAP of my 1-class model:
`darknet/darknet detector map data/training/rapidus-1.data data/models/rapidus-1.cfg data/models/rapidus-1.weights`

## Troubleshooting
- If you get a python error "no module named _tkinter": `apt-get install python3-tk`
- If you get an error during build of darknet "#include <opencv2/highgui.hpp> not found": change path to opencv2/highgui/highgui.hpp and try building again

## Acknowledgement
Joseph Chet Redmon et al. for writing the deep learning framework [darknet](https://pjreddie.com/darknet/)
AlexeyAB for making darknet [available for windows](https://github.com/AlexeyAB/darknet)
duangenquan for [converting YoloV2 to NCS](https://github.com/duangenquan/YoloV2NCS)