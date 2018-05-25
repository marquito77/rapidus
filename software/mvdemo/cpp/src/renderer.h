#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <linux/fb.h>

class Renderer
{
protected:
    Renderer() {};
    virtual ~Renderer() {};
public:
    virtual bool render(const cv::Mat& i_img) = 0;
};

class OpenCvRenderer : public Renderer
{
public:
    explicit OpenCvRenderer(const std::string& i_windowName);
    virtual ~OpenCvRenderer();
    virtual bool render(const cv::Mat& i_img);

private:
    std::string m_windowName;
};

class FramebufferRenderer : public Renderer
{
public:
    FramebufferRenderer();
    virtual ~FramebufferRenderer();
    virtual bool render(const cv::Mat& i_img);

private:

	struct fb_fix_screeninfo m_currFinfo;
	struct fb_var_screeninfo m_oldVinfo;
	struct fb_var_screeninfo m_currVinfo;

	int m_fbFd;
	char* m_fbPtr;
	bool m_isInit;

	bool init(int i_width, int i_height, int i_bitsPerPixel);
};
