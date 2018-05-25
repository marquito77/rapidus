
#include "region.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>

//#define TRACE_BOXES


Region::Region()
{
	initialized = false;
}


bool Region::Initialize(int layerWidth, int layerHeight, int numClasses,
                        float _nms, const std::vector<float>& _biases)
{
	size = 4 + numClasses + 1;
	N = 5;
	c = (numClasses+5) * 5;
	h = layerHeight;
	w = layerWidth;
	classes = numClasses;
	nms = _nms;
	blockwd = layerWidth;

	if ((int)_biases.size()/2 != N) {
		printf("ERROR: Number of anchor coordinates must be %d (is %d)\n", N, (int)_biases.size());
		return false;
	}

	biases = _biases;

	totalLength = c * h * w;
	totalObjects = N * h * w;
	output.resize(totalLength);
	boxes.resize(totalObjects);
	s.resize(totalObjects);

	for(int i = 0; i < totalObjects; ++i)
	{
		s[i].index = i;
		s[i].channel = size;
		s[i].prob = &output[5];
	}

	initialized = true;
	return true;
}


void Region::GetDetections(float* data, float thresh, std::vector<DetectedObject> &objects)
{
#ifdef TRACE_BOXES
    FILE* fh = fopen("boxes.txt", "w");
    FILE* fhMov = fopen("movout.txt", "w");
    fprintf(fh, "idx, scale, x, y, w, h\n");
#endif

	objects.clear();

	if(!initialized)
	{
		printf("Fail to initialize internal buffer!\n");
		return ;
	}

	if ((int)biases.size() != 2*N) 
	{
		printf("ERROR: vector size of biases must be 2*N! (N = %d, biases.size() = %d)\n",
			N, (int)biases.size());
		return;
	}

	int i,j,k;

#ifdef TRACE_BOXES
	for (int i=0; i<w*h*N*size; ++i) {
		fprintf(fhMov, "%f\n", output[i]);
	}
#endif	

	//transpose(data, &output[0], size*N, w*h);
	output.assign(data, data+output.size());

	// Initialize box, scale and probability
	for(i = 0; i < h*w*N; ++i)
	{
		int index = i * size;
		//Box
		int n = i % N;
		int row = (i/N) / w;
		int col = (i/N) % w;

		boxes[i].x = (col + logistic_activate(output[index + 0])) / blockwd; //w;
		boxes[i].y = (row + logistic_activate(output[index + 1])) / blockwd; //h;
		boxes[i].w = exp(output[index + 2]) * biases[2*n]   / blockwd; //w;
		boxes[i].h = exp(output[index + 3]) * biases[2*n+1] / blockwd; //h;

		//Scale
		output[index + 4] = logistic_activate(output[index + 4]);

#ifdef TRACE_BOXES
		fprintf(fh, "%d, %f, %f, %f, %f, %f\n", 
			i, output[index + 4], boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
#endif

		//Class Probability
		softmax(&output[index + 5], classes, 1, &output[index + 5]);
		for(j = 0; j < classes; ++j)
		{
			output[index+5+j] *= output[index+4];
			if(output[index+5+j] < thresh) output[index+5+j] = 0;
		}
	}
#ifdef TRACE_BOXES	
	fclose(fh);
	fclose(fhMov);
#endif	

	//nms
	for(k = 0; k < classes; ++k)
	{
		for(i = 0; i < totalObjects; ++i)
		{
			s[i].iclass = k;
		}
		qsort(&s[0], totalObjects, sizeof(indexsort), indexsort_comparator);
		for(i = 0; i < totalObjects; ++i){
			if(output[s[i].index * size + k + 5] == 0) continue;
			ibox a = boxes[s[i].index];
			for(j = i+1; j < totalObjects; ++j){
				ibox b = boxes[s[j].index];
				if (box_iou(a, b) > nms){
					output[s[j].index * size + 5 + k] = 0;
				}
			}
		}
	}

	// generate objects
	for(i = 0, j = 5; i < totalObjects; ++i, j += size)
	{
		int iclass = max_index(&output[j], classes);

		float prob = output[j+iclass];

		if(prob > thresh)
		{
			ibox b = boxes[i];

			//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

			float left  = (b.x-b.w/2.);
			float right = (b.x+b.w/2.);
			float top   = (b.y-b.h/2.);
			float bot   = (b.y+b.h/2.);

			if(left < 0.) left = 0.;
			if(right > 1.) right = 1.;
			if(top < 0.) top = 0.;
			if(bot > 1.) bot = 1.;


			DetectedObject obj;
			obj.left = left;
			obj.top = top;
			obj.right = right;
			obj.bottom = bot;
			obj.confidence = prob;
			obj.objType = iclass;
			//obj.name = objectnames[iclass];
			objects.push_back(obj);
		}
	}

	return ;
}

