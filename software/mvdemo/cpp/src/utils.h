#pragma once

#include <vector>
#include <string>

#define YOLO_CONFIG_DEFAULT_OUTPUT_WIDTH  13
#define YOLO_CONFIG_DEFAULT_NMS           0.4f

struct YoloConfig {
    int inWidth;
    int inHeight;
    int inChannels;
    int outWidth;
    int outHeight;
    int outChannels;
    int numClasses;
    int numBoxes;
    float nms;
    std::vector<float> anchors;
};
bool readYoloConfig(const char* i_filename, 
                    YoloConfig& o_config, 
                    float i_nms=YOLO_CONFIG_DEFAULT_NMS, 
                    int i_outputWidth=YOLO_CONFIG_DEFAULT_OUTPUT_WIDTH);

bool readNamesFile(const std::string& i_filename, std::vector<std::string>& o_classNames);