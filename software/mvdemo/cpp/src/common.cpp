#include "common.h"
#include <sstream>
#include <fstream>

#include <sys/stat.h>


int indexsort_comparator(const void *pa, const void *pb)
{
    float proba = ((indexsort *)pa)->prob[((indexsort *)pa)->index * ((indexsort *)pa)->channel + ((indexsort *)pa)->iclass];
    float probb = ((indexsort *)pb)->prob[((indexsort *)pb)->index * ((indexsort *)pb)->channel + ((indexsort *)pb)->iclass];

    float diff = proba - probb;
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float logistic_activate(float x)
{
    return 1./(1. + exp(-x));
}

void transpose(float *src, float* tar, int k, int n)
{
    int i, j, p;
    float *tmp = tar;
    for(i = 0; i < n; ++i)
    {
        for(j = 0, p = i; j < k; ++j, p += n)
        {
            *(tmp++) = src[p];
        }
    }
}
void softmax(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
}
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(ibox a, ibox b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(ibox a, ibox b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(ibox a, ibox b)
{
    return box_intersection(a, b)/box_union(a, b);
}
int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

double getTimeMs()
{
    struct timespec tval;
    clock_gettime(CLOCK_MONOTONIC_RAW, &tval);

    long sec  = tval.tv_sec;
    long usec = tval.tv_nsec/1000;

    double ds = (double)sec*1000. + (double)usec / 1000.;

    return ds;
}

void* loadGraphFile(const char* i_filename, uint32_t* o_buffLen)
{
    *o_buffLen = 0;
    char* buff = NULL;
    FILE* fh = fopen(i_filename, "rb");
    if (fh != NULL) {
        fseek(fh , 0 , SEEK_END);
        long int lSize = ftell(fh);
        rewind(fh);
        buff = (char*)malloc(sizeof(char)*lSize);
        if (NULL == buff) {
            printf("ERROR: Out of Memory while reading file %s\n", i_filename);
        } else {
            size_t bytesRead = fread(buff, 1, lSize, fh);
            if (bytesRead != (size_t)lSize) {
                printf("ERROR: Could not fully read file %s\n", i_filename);
                free(buff);
                buff = NULL;
            } else {
                *o_buffLen = lSize;
            }
        }

        fclose(fh);
    } else {
        printf("ERROR: Could not open file %s\n", i_filename);
    }

    return (void*)buff;
}

void rescaleDetection(DetectedObject& io_obj, const ScaleData& i_scale)
{
    io_obj.left   = (int)(io_obj.left   * i_scale.imgW/i_scale.scaleX - i_scale.offX);
    io_obj.right  = (int)(io_obj.right  * i_scale.imgW/i_scale.scaleX - i_scale.offX);
    io_obj.top    = (int)(io_obj.top    * i_scale.imgH/i_scale.scaleY - i_scale.offY);
    io_obj.bottom = (int)(io_obj.bottom * i_scale.imgH/i_scale.scaleY - i_scale.offY);
}

std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

std::string trim(const std::string& s)
{
    std::string ret = s;

    // right trim
    size_t pos = ret.find_last_not_of(" \n\r\t");
    if (pos != std::string::npos) {
        ret.erase(pos+1);
    }

    // left trim
    pos = ret.find_first_not_of(" \n\r\t");
    if (pos != std::string::npos) {
        ret = ret.substr(pos);        
    }
    
    return ret;
}

void getValsFromCfg(const char* i_filename, const char* i_section, const char* i_key, 
                   std::vector<float>& o_vals)
{
    o_vals.clear();

    std::ifstream infile(i_filename);
    if (!infile.is_open()) {
        printf("ERROR: Could not open config file %s\n", i_filename);
        return;
    }

    std::string line;
    bool inSection = false;
    while (std::getline(infile, line)) {
        line = trim(line);
        size_t commentPos = line.find_first_of("#");
        if (commentPos != std::string::npos) {
            line.erase(commentPos);            
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] == '[') {
            if (line.find(i_section) != std::string::npos) {
                inSection = true;
            }
            continue;
        }
        if (inSection) {
            if (line.find(i_key) != 0) {
                continue;
            }
            std::vector<std::string> tokens = split(line, '=');
            if (tokens.size() != 2) {
                printf("ERROR: Could not parse line in config file: %s\n", line.c_str());
                return;
            }
            std::string valStr = trim(tokens[1]);
            tokens = split(valStr, ',');
            for (int i=0; i<(int)tokens.size(); ++i) {
                std::string tok = tokens[i];
                tok = trim(tok);
                float val = stof(tok);
                o_vals.push_back(val);
            }

            return;
        }
    }

    infile.close();
}

bool checkFile(const std::string& i_filename)
{
    struct stat buffer;
    bool fileExists = stat(i_filename.c_str(), &buffer) == 0;
    
    if (!fileExists) {
        printf("Could not access file %s\n", i_filename.c_str());
    }

    return fileExists;
}
