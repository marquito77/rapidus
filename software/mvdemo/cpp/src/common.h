#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <string>
#include <stdio.h>

struct DetectedObject
{
    float left, top, right, bottom;
    float confidence;
    int objType;
    std::string className;
};

struct ScaleData {
    float imgW;
    float imgH;
    float offX;
    float offY;
    float scaleX;
    float scaleY;
};

struct ibox
{
    float x, y, w, h;
};

struct indexsort
{
    int iclass;
    int index;
    int channel;
    float* prob;
};

int indexsort_comparator(const void *pa, const void *pb);

float logistic_activate(float x);
void transpose(float *src, float* tar, int k, int n);
void softmax(float *input, int n, float temp, float *output);
float overlap(float x1, float w1, float x2, float w2);
float box_intersection(ibox a, ibox b);
float box_union(ibox a, ibox b);
float box_iou(ibox a, ibox b);
int max_index(float *a, int n);

double getTimeMs();
void* loadGraphFile(const char* i_filename, uint32_t* o_buffLen);
void rescaleDetection(DetectedObject& io_obj, const ScaleData& i_scale);

std::vector<std::string> split(const std::string& s, char delimiter);
std::string trim(const std::string& s);

void getValsFromCfg(const char* i_filename, const char* i_section, const char* i_key,
	               std::vector<float>& o_vals);

template<typename T>
bool getValFromCfg(const char* i_filename, const char* i_section, const char* i_key, T& o_val)
{
	std::vector<float> vec;
	getValsFromCfg(i_filename, i_section, i_key, vec);
	if (vec.size() != 1) {
		return false;
	}

	o_val = (T)vec[0];

	return true;
}

bool checkFile(const std::string& i_filename);