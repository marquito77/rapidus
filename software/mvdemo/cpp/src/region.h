#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>

#include "common.h"



struct Region
{
    bool initialized;
    int totalLength;
    int totalObjects;
    std::vector<float> output;
    std::vector<ibox> boxes;
    std::vector<indexsort> s;
    std::vector<float> biases;

    int c, h, w, classes, blockwd, size, N;
    float nms;

    Region();

    // 
    bool Initialize(int layerWidth, int layerHeight, int numClasses,
                    float nms, const std::vector<float>& biases);


    void GetDetections(float* data, float thresh, std::vector<DetectedObject> &objects);
};

