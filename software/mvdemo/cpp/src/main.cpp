#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <thread>

#include <opencv2/opencv.hpp>
#include <mvnc.h>

#include <iostream>

#include "floatcompressor.h"
#include "region.h"
#include "common.h"
#include "sharedBuffer.h"
#include "utils.h"
#include "renderer.h"

using namespace cv;

#define SHARED_BUFFER_SIZE 2
#define DEFAULT_THRESH 0.25f
#define INVALID_THRESH -1.f

struct MainArgs
{
    std::string configFile;
    std::string graphFile;
    std::string namesFile;
    std::string sourceFile;
    std::string outputFile;
    float       thresh;
    std::vector<std::string> classNames;
};

static YoloConfig g_yoloConfig;
static MainArgs   g_mainArgs;
static Renderer*  g_renderer;

static void* g_hMov;
static void* g_hGraph;
static void* g_graphFileBuf;
static Region g_regionLayer;
static ScaleData g_scaleData;
bool g_done;
bool g_isInitScaleData;
std::vector<float32_t> g_resultsBufF32;

static SharedBuffer<Mat>                 g_imgSharedBuffer(SHARED_BUFFER_SIZE);
static SharedBuffer<Mat>                 g_imgScaledSharedBuffer(SHARED_BUFFER_SIZE);
static SharedBuffer<std::vector<half_t>> g_resultsSharedBuffer(SHARED_BUFFER_SIZE);

void usage()
{
    printf("\n");
    printf("Usage: mvdetect <imagesource> [-cngt]\n");
    printf("    <imagesource>: This can be a path to a video/image file or a webcam device\n");
    printf("    Note: You must specify at least one of the files (options cng).\n");
    printf("          Config, graph and names files must be in the same directory and\n");
    printf("          have the same basename (e.g.: test.cfg, test.graph, test.names).\n");
    printf("    -c: config file.\n");
    printf("        This is the darknet .cfg file which contains the model description.\n");
    printf("    -n: names file\n");
    printf("        This is the darknet .names file which\n");
    printf("        contains the list of trained classes.\n");
    printf("    -g: graph file\n");
    printf("        This is the movidius .graph file which\n");
    printf("        contains the model and weights for the NCS.\n");
    printf("    -o: output file\n");
    printf("        Processed images, videos or streams are saved into this file.\n");
    printf("    -t: threshold\n");
    printf("        Threshold for accepting/dismissing classification result.\n");
    printf("        It must be a floating point number between 0 and 1.\n");
    printf("\n");

    exit(-1);
}


static bool initArgs()
{
    if (g_mainArgs.sourceFile.empty()) {
        return false;        
    }

    bool hasConfig = !g_mainArgs.configFile.empty();
    bool hasGraph  = !g_mainArgs.graphFile.empty();
    bool hasNames  = !g_mainArgs.namesFile.empty();

    if (!hasConfig) {
        if (hasGraph || hasNames) {
            std::string s;
            if (hasGraph) {
                s = g_mainArgs.graphFile;
            } else {
                s = g_mainArgs.namesFile;
            }
            size_t posExt = s.rfind('.');
            if (posExt == std::string::npos) {
                posExt = s.size();
            }
            std::string sConf = s.substr(0, posExt);
            sConf += ".cfg";
            g_mainArgs.configFile = sConf;
            hasConfig = true;
        } else {
            return false;
        }
    }

    // from this point on we must have a config filename
    if (g_mainArgs.configFile.empty()) {
        return false;
    }

    size_t pos = g_mainArgs.configFile.rfind('.');
    const std::string basename = g_mainArgs.configFile.substr(0, pos);
    if (!hasGraph) {        
        g_mainArgs.graphFile = basename + ".graph";
        hasGraph = true;
    }

    if (!hasNames) {
        g_mainArgs.namesFile = basename + ".names";
        hasNames = true;
    }

    if (g_mainArgs.thresh == INVALID_THRESH) {
        g_mainArgs.thresh = DEFAULT_THRESH;
    }

    if (   !checkFile(g_mainArgs.configFile)
        || !checkFile(g_mainArgs.graphFile)) {
        return false;
    }

    return true;
}

static bool init()
{
    if (!initArgs()) {
        printf("Error while initializing arguments\n");
        usage();
        return false;
    }

    if (!readYoloConfig(g_mainArgs.configFile.c_str(), g_yoloConfig)) {
        printf("Could not read yolo config file %s\n", g_mainArgs.configFile.c_str());
        return false;
    }

    if (!readNamesFile(g_mainArgs.namesFile, g_mainArgs.classNames)) {
        printf("Warning: .names file does not exist or could not be read: %s\n", g_mainArgs.namesFile.c_str());
    }

    // check if number of classnames fit to yolo config
    if (g_yoloConfig.numClasses != (int)g_mainArgs.classNames.size()) {
        printf("Warning: #classes in config (=%d) does not fit to #names in .names file (=%d). "
               "Class names will be created (class0, class1...) or removed\n",
               g_yoloConfig.numClasses, (int)g_mainArgs.classNames.size());
        if (g_yoloConfig.numClasses < (int)g_mainArgs.classNames.size()) {
            g_mainArgs.classNames.resize(g_yoloConfig.numClasses);
        } else {
            for(int i=g_mainArgs.classNames.size(); i<g_yoloConfig.numClasses; ++i) {
                std::string cn("class");
                cn += std::to_string(i);
                g_mainArgs.classNames.push_back(cn);
            }
        }
    }

    bool ok = g_regionLayer.Initialize(g_yoloConfig.outWidth, g_yoloConfig.outHeight, 
                                    g_yoloConfig.numClasses, g_yoloConfig.nms, g_yoloConfig.anchors);

    // if we are in a xwindow environment we can use opencv gui
    // else we draw directly into the framebuffer - yeah!
    int ret = std::system("xset -q >/dev/null 2>&1");
    if (ret == 0) {
        g_renderer = new OpenCvRenderer(g_mainArgs.sourceFile);
    } else {
        g_renderer = new FramebufferRenderer();
    }

    return ok;
}

static bool initMov()
{
    g_hMov = NULL;
    g_graphFileBuf = NULL;

    bool ret = true;
    mvncStatus retCode;
    char devName[100];

    int loglvl = 1;
    mvncSetGlobalOption(MVNC_LOG_LEVEL, &loglvl, sizeof(loglvl));

    retCode = mvncGetDeviceName(0, devName, sizeof(devName));
    if (retCode != MVNC_OK)
    {   // If failed to get device name, may be none plugged in.
        printf("No NCS devices found\n");
        goto error;
    }
    
    // Try to open the NCS device via the device name.
    retCode = mvncOpenDevice(devName, &g_hMov);

    if (retCode != MVNC_OK)
    {   // Failed to open the device.  
        printf("Could not open NCS device with error = %d\n", retCode);
        goto error;
    }
    
    // deviceHandle is ready to use now.  
    // Pass it to other NC API calls as needed, and close it when finished.
    printf("Successfully opened NCS device!\n");

    uint32_t graphFileLen;
    g_graphFileBuf = loadGraphFile(g_mainArgs.graphFile.c_str(), &graphFileLen);
    if (NULL == g_graphFileBuf) {
        goto error;
    } else {
        printf("Successfully loaded graph file %s\n", g_mainArgs.graphFile.c_str());
    }

    retCode = mvncAllocateGraph(g_hMov, &g_hGraph, g_graphFileBuf, graphFileLen);
    if (retCode != MVNC_OK) {
        printf("ERROR: Could not allocate graph\n");
        goto error;
    } else {
        printf("Successfully allocated graph\n");
    }

    goto ok;

error:
    if (g_hMov != NULL) {
        mvncCloseDevice(g_hMov);
        g_hMov = NULL;
    }
    ret = false;

ok:
    return ret;
}

static bool preprocess(const Mat& i_img, Mat& o_img, ScaleData& o_scale)
{
    int origW = i_img.cols;
    int origH = i_img.rows;

    int max_dim = (origW >= origH) ? origW : origH;
    float scale = (float)g_yoloConfig.inWidth / max_dim;
    Rect roi;
    if (origW >= origH) {
        roi.x = 0;
        roi.width = g_yoloConfig.inWidth;
        roi.height = origH * scale;
        roi.y = (g_yoloConfig.inHeight - roi.height) / 2;
    } else {
        roi.y = 0;
        roi.height = g_yoloConfig.inHeight;
        roi.width = origW * scale;
        roi.x = (g_yoloConfig.inWidth - roi.width) / 2;
    }

    Mat imgScaled = Mat::zeros(g_yoloConfig.inWidth, g_yoloConfig.inHeight, i_img.type());
    resize(i_img, imgScaled(roi), roi.size(), 0, 0, INTER_LINEAR);
    imgScaled.convertTo(imgScaled, CV_32FC3, 1./255.);
    cvtColor(imgScaled, imgScaled, COLOR_RGB2BGR);
    //resize(i_img, o_img, Size(g_dim, g_dim), 0, 0, INTER_LINEAR);

    o_img = Mat::zeros(g_yoloConfig.inWidth, g_yoloConfig.inHeight, CV_16UC3);

    int numPix = imgScaled.total()*imgScaled.channels();
    floats2halfs(imgScaled.ptr<float>(), o_img.ptr<half_t>(), numPix);

    o_scale.imgW = origW;
    o_scale.imgH = origH;
    o_scale.offX = ((float)roi.x*origW) / roi.width;
    o_scale.offY = ((float)roi.y*origH) / roi.height;
    o_scale.scaleX = (float)roi.width/g_yoloConfig.inWidth;
    o_scale.scaleY = (float)roi.height/g_yoloConfig.inHeight;

#if 0
    printf("%f, %f, %f, %f, %f, %f\n",
        o_scale.imgW, o_scale.imgH, o_scale.offX, o_scale.offY, o_scale.scaleX, o_scale.scaleY);
#endif        

    return true;
}

static bool preprocess2(const Mat& i_img, Mat& o_img, uint16_t* o_imgF16, ScaleData& o_scale)
{
    resize(i_img, o_img, Size(g_yoloConfig.inWidth, g_yoloConfig.inHeight), 0, 0, INTER_LINEAR);

    o_img.convertTo(o_img, CV_32FC3, 1./255.);
    cvtColor(o_img, o_img, COLOR_RGB2BGR);

    int numPix = o_img.total()*o_img.channels();

    floats2halfs(o_img.ptr<float>(), o_imgF16, numPix);

    o_scale.imgW = i_img.cols;
    o_scale.imgH = i_img.rows;
    o_scale.offX = 0.f;
    o_scale.offY = 0.f;
    o_scale.scaleX = 1.f;
    o_scale.scaleY = 1.f;

    return true;
}

static bool detect(const Mat& i_img, std::vector<half_t>& o_results)
{
    void* userParams;
    const void* pTensor = (const void*)i_img.ptr<half_t>();
    uint32_t tensorSize = i_img.total() * i_img.channels() * sizeof(half_t);
    mvncStatus ret = mvncLoadTensor(g_hGraph, pTensor, tensorSize, NULL);
    if (MVNC_OK != ret) {
        printf("ERROR: Could not load tensor\n");
        return false;
    }

    void* results;
    uint32_t resultsLen;
    ret = mvncGetResult(g_hGraph, &results, &resultsLen, &userParams);
    if (MVNC_OK != ret) {
        printf("ERROR: Could not get results from MNCS\n");
        return false;
    }

    uint32_t resSize = resultsLen/sizeof(half_t);
    o_results.resize(resSize);

    o_results.assign((half_t*)results, (half_t*)results + resSize);

    return true;
}

static bool postprocess(const std::vector<half_t>& i_results, std::vector<DetectedObject>& o_detections)
{    
    g_resultsBufF32.resize(i_results.size());
    halfs2floats(&i_results[0], g_resultsBufF32.data(), i_results.size());

    g_regionLayer.GetDetections(g_resultsBufF32.data(), g_mainArgs.thresh, o_detections);

    return true;
}

static void visualize(const Mat& i_img, Mat& o_img, 
                      const std::vector<DetectedObject>& i_detections, 
                      const ScaleData& i_scale,
                      double fps)
{
    //colors = [(128,128,128),(128,0,0),(192,192,128),(255,69,0),(128,64,128),(60,40,222),(128,128,0),(192,128,128),(64,64,128),(64,0,128),(64,64,0),(0,128,192),(0,0,0)];

    o_img = i_img.clone();
    //printf("imgW=%d imgH=%d\n", imgW, imgH);
    Scalar whiteColor(255, 255, 255, 255);
    Scalar blackColor(0, 0, 0, 255);
    char buff[64];
    snprintf(buff, sizeof(buff), "%.1f", fps);
    putText(o_img, buff, Point(o_img.cols-50, 20), FONT_HERSHEY_SIMPLEX, 0.5, whiteColor);
    for (int i=0; i<(int)i_detections.size(); ++i) {
        DetectedObject det = i_detections[i];
        Color clr = getColor(det.objType, g_yoloConfig.numClasses);
        Scalar boxColor(clr.r, clr.g, clr.b, 255);
        rescaleDetection(det, i_scale);
        const char* className = g_mainArgs.classNames[det.objType].c_str();

        Rect r1(Point(det.left, det.top), Point(det.right, det.bottom));
        //printf("%d: bboxRect = (%d, %d, %d, %d)\n", i, r1.x, r1.y, r1.x+r1.width, r1.y+r1.height);
        Rect r2 = Rect(r1.x, r1.y-20, r1.width, 20);
        rectangle(o_img, r1, boxColor, 3);
        rectangle(o_img, r2, boxColor, CV_FILLED);

        putText(o_img, className, Point(r1.x+5, r1.y-7), FONT_HERSHEY_SIMPLEX, 0.5, blackColor, 2);
    }
}

class ImageWriter
{
public:
    ImageWriter(const std::string& i_source, const std::string& o_output)
    : m_filename(o_output)
    {
        //TODO: determine if source is image
    }

    void write(const Mat& i_img)
    {
        if (m_sourceIsImage) {
            imwrite(m_filename, i_img);
        } else {
            if (!m_videoWriter.isOpened()) {
                int fourcc = CV_FOURCC('D', 'I', 'V', 'X');
                Size frameSize(i_img.rows, i_img.cols);
                m_videoWriter.open(o_output, fourcc, 20, frameSize, 1);                
            }

            if (m_videoWriter.isOpened()) {
                m_videoWriter.write(i_img)
            }
        }
    }

private:
    bool m_sourceIsImage;
    std::string m_filename;
    VideoWriter m_videoWriter;
};

static void saveImage(const Mat& i_img, const std::string& i_source, const std::string& i_output)
{

}

void thrdProvideImage()
{
    VideoCapture cap(g_mainArgs.sourceFile);
    Mat img;
    Mat imgCpy;
    Mat imgScaled;
    while (!g_done) {
        cap.read(img);
        //imgCpy = img.clone();
        g_imgSharedBuffer.put(img);

        preprocess(img, imgScaled, g_scaleData);

        g_imgScaledSharedBuffer.put(imgScaled);
    }
}

void thrdDetect()
{
    Mat scaledImg;
    std::vector<half_t> results;
    while (!g_done) {
        scaledImg = g_imgScaledSharedBuffer.get();
        detect(scaledImg, results);
        g_resultsSharedBuffer.put(results);
    }
}

bool parseArgs(int argc, char** argv)
{
    int c;
    g_mainArgs.thresh = INVALID_THRESH;
    while ((c = getopt(argc, argv, "c:g:n:o:t:")) != -1) {
        switch (c)
        {
            case 'c':
                g_mainArgs.configFile = optarg;
                break;
            case 'g':
                g_mainArgs.graphFile = optarg;
                break;
            case 'n':
                g_mainArgs.namesFile = optarg;
                break;
            case 'o':
                g_mainArgs.outputFile = optarg;
                break;
            case 't':
                g_mainArgs.thresh = atof(optarg);
                break;
            default:
                usage();
        }
    }

    for (int index = optind; index < argc; index++) {
        g_mainArgs.sourceFile = argv[index];
    }

    return true;
}

int main(int argc, char** argv)
{
    // parse arguments
    if (!parseArgs(argc, argv)) {
        printf("Error while parsing arguments\n");
        return -1;
    }

    if (!init()) {
        printf("ERROR: could not initalize program\n");
        return -1;
    }

    if (!initMov()) {
        printf("ERROR: could not initialize Movidius NCS\n");
        return -1;
    }
    //namedWindow(g_imageSource, CV_WINDOW_AUTOSIZE);

//    Mat img;
    Mat imgVis;
//    Mat imgScaled;
    std::vector<half_t> results;
    //ScaleData scaleData;

    VideoCapture cap(g_mainArgs.sourceFile);

    bool ok = true;
    int numFrames = 0;
    double avrgTime = 0.05;
    double alpha = 0.05;
    double fps = 0.;
    std::vector<DetectedObject> detections;

    std::thread t1(thrdProvideImage);
    std::thread t2(thrdDetect);

    printf("Entering main loop...\n");
    double ts = getTimeMs();
    while(1) {
        double tStart = getTimeMs();

        const std::vector<half_t>& results = g_resultsSharedBuffer.get();
        ok = postprocess(results, detections);
        if (!ok) {
            break;
        }
        const Mat& img = g_imgSharedBuffer.get();
        visualize(img, imgVis, detections, g_scaleData, fps);

        if (!g_mainArgs.outputFile.empty()) {
            saveImage(imgVis, false);
        }
        
        //double tRend = getTimeMs();
        ok = g_renderer->render(imgVis);
        if (!ok) {
            break;
        }

		//printf("Time render: %.2fms\n", getTimeMs()-tRend);
        numFrames += 1;
        double t = getTimeMs();
        double tDiff = t - tStart;
        avrgTime += alpha*(tDiff - avrgTime);
        if ((t-ts) >= 1000.) {
            fps = (1000.*numFrames)/(t-ts);
            //printf("Avrg time: %.2fms, fps: %.1f\n", avrgTime, fps);
            ts = t;
            numFrames = 0;
        }

    }
    printf("Exit main loop\n");

    // signal threads to finish
    g_done = true;

    // threads might be busy waiting for queues so we have to wake them up by emptying queues
    g_imgSharedBuffer.clear();
    g_imgScaledSharedBuffer.clear();
    g_resultsSharedBuffer.clear();

    t1.join();
    g_imgSharedBuffer.clear();
    g_imgScaledSharedBuffer.clear();
    g_resultsSharedBuffer.clear();
    t2.join();

    if (NULL != g_graphFileBuf) {
        free(g_graphFileBuf);
        g_graphFileBuf = NULL;
    }

    if (g_hGraph != NULL) {
        mvncDeallocateGraph(g_hGraph);
        g_hGraph = NULL;
    }

    if (g_hMov != NULL) {
        mvncCloseDevice(g_hMov);
        g_hMov = NULL;      
    }

    return 0;
}
