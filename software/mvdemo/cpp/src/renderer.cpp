#include "renderer.h"

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>


FramebufferRenderer::FramebufferRenderer()
: m_fbFd(-1)
, m_fbPtr(NULL)
, m_isInit(false)
{
	memset(&m_currFinfo, 0, sizeof(m_currFinfo));
	memset(&m_oldVinfo,  0, sizeof(m_oldVinfo));
	memset(&m_currVinfo, 0, sizeof(m_currVinfo));

	const char* fbdev = "/dev/fb0";
	m_fbFd = open(fbdev, O_RDWR);
	if (m_fbFd == -1) {
		printf("Could not open framebuffer device %s\n", fbdev);
		return;
	}

	if (ioctl(m_fbFd, FBIOGET_VSCREENINFO, &m_oldVinfo)) {
		printf( "Unable to read variable screen information\n");
		return;
	} else {
		m_currVinfo = m_oldVinfo;
		printf("Original vscreen info: vxres = %d, vyres = %d, bpp = %d\n", 
			   m_oldVinfo.xres, m_oldVinfo.yres, m_oldVinfo.bits_per_pixel);
	}
}

FramebufferRenderer::~FramebufferRenderer()
{
	if (m_fbFd != -1) {
		if (m_oldVinfo.bits_per_pixel != 0) {
			if (!ioctl(m_fbFd, FBIOPUT_VSCREENINFO, &m_oldVinfo)) {
				printf("Could not set framebuffer config to original values\n");
			}			
		}
		close(m_fbFd);
		m_fbFd = -1;
	}

	if (m_fbPtr != NULL) {
		munmap(m_fbPtr, m_currFinfo.smem_len);
		m_fbPtr = NULL;	
	}
}

bool FramebufferRenderer::init(int i_width, int i_height, int i_bitsPerPixel)
{
	m_isInit = false;
	if (m_fbFd == -1) {
		return false;
	}

	m_currVinfo.bits_per_pixel = i_bitsPerPixel;
	m_currVinfo.xres = i_width;
	m_currVinfo.yres = i_height;
	m_currVinfo.xres_virtual = i_width;
	m_currVinfo.yres_virtual = i_height;
	m_currVinfo.xoffset = 0;
	m_currVinfo.yoffset = 0;
	if (ioctl(m_fbFd, FBIOPUT_VSCREENINFO, &m_currVinfo)) {
  		printf("Error setting variable information.\n");
  		return false;
	} else {
		printf("Setting framebuffer to image size: vxres = %d, vyres = %d, bpp = %d\n",
			m_currVinfo.xres, m_currVinfo.yres, m_currVinfo.bits_per_pixel);
	}

	if (ioctl(m_fbFd, FBIOGET_FSCREENINFO, &m_currFinfo)) {
		printf("Unable to read fixed screen information\n");
		return false; 
	} else {
		printf("smem_len = %d, line_length=%d\n", 
			   m_currFinfo.smem_len, m_currFinfo.line_length);
	}

	m_fbPtr = (char*)mmap(0, 
                m_currFinfo.smem_len, 
                PROT_READ | PROT_WRITE, 
                MAP_SHARED, 
                m_fbFd, 0);
    if (m_fbPtr == MAP_FAILED) {
		printf("Could not allocate memory for framebuffer\n");
		return false;
	}


	m_isInit = true;
	return true;
}

bool FramebufferRenderer::render(const cv::Mat& i_img)
{
	if (!m_isInit) {
		if (!init(i_img.cols, i_img.rows, 32)) {
			return false;
		}
	}

	uint8_t* data;
	static cv::Mat frame(i_img.rows, i_img.cols, CV_8UC4);
    if (   frame.rows != i_img.rows
    	|| frame.cols != i_img.cols) {
    	frame = cv::Mat(i_img.rows, i_img.cols, CV_8UC4);
    }

	cv::cvtColor(i_img, frame, cv::COLOR_BGR2BGRA);
	data = frame.data;
#if 1
	memcpy(m_fbPtr, data, m_currFinfo.smem_len);
#else
	for (int r=0; r<720; ++r) {
		for (int c=0; c<1280; ++c) {
			int idx = r*finfo.line_length + c*4;
			Vec4b pix = frame.at<Vec4b>(r,c);
			fbp[idx+0] = pix[0];
			fbp[idx+1] = pix[1];
			fbp[idx+2] = pix[2];
			fbp[idx+3] = pix[3];
		}
	}
#endif
	(void)data;

  return true;
}

OpenCvRenderer::OpenCvRenderer(const std::string& i_windowName)
: m_windowName(i_windowName)
{
	cv::namedWindow(m_windowName, CV_WINDOW_AUTOSIZE);
}

OpenCvRenderer::~OpenCvRenderer()
{
	cv::destroyWindow(m_windowName);
}

bool OpenCvRenderer::render(const cv::Mat& i_img)
{
	cv::imshow(m_windowName, i_img);
	char c = cv::waitKey(1);
	if (c==27 || c=='q') {
		return false;
	} else {
		return true;
	}
}