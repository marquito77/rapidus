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
- Full installation of Movidius SDK with OpenCV

## Demo (using NCS)
To run the demo video on the Movidius NCS first compile the c++ application:
```
cd mvdemo/cpp
make
cd ../..
```
Connect the NCS and run the demo:
`sh runDemoCpp.sh`
This should work on a PC/Linux as well as on the Raspberry Pi.

## Demo (using darknet, not NCS)
To test my 10-class model on one image:
`darknet/darknet detector test data/training/model_10class.data data/models/model_10class.cfg data/models/model_10class.weights data/media/perTieCell.jpg`
To test my 1-class model on video:
`darknet/darknet detector demo data/training/model_1class.data data/models/model_1class.cfg data/models/model_1class.weights data/media/mall.mp4`

## Training
If you want to reproduce the training as described in my report you need to do the following from root directory:
1. download and build the darknet fork of Alexey 
`sh runDownloadDarknet`
2. download COCO
`python runDownloadCoco.py`
3. create COCO datasets
`python runCreateDatabase.py`
4. start training
`darknet/darknet detector train data/training/model_1class.data data/models/model_1class.cfg`
Training should run very slow because GPU and CUDA support in darknet is turned off by default. Edit darknet/Makefile and turn on 
GPU=1, CUDA=1 and OPENCV=1

## Validation
To calculate the mAP of my 1-class model:
`darknet/darknet detector map data/training/model_1class.data data/models/model_1class.cfg data/models/model_1class.weights`

## Acknowledgement
Joseph Chet Redmon et al. for writing the deep learning framework [darknet](https://pjreddie.com/darknet/)
AlexeyAB for making darknet [available for windows](https://github.com/AlexeyAB/darknet)
duangenquan for [converting YoloV2 to NCS](https://github.com/duangenquan/YoloV2NCS)