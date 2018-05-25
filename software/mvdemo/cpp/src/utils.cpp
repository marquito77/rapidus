#include "utils.h"
#include "common.h"
#include <fstream>
#include <stdlib.h>

bool readYoloConfig(const char* i_filename, YoloConfig& o_config, float i_nms, int i_outputWidth)
{
	if (!getValFromCfg(i_filename, "net", "width", o_config.inWidth)) {
        return false;
    }

    if (!getValFromCfg(i_filename, "net", "height", o_config.inHeight)) {
        return false;
    }

    if (!getValFromCfg(i_filename, "net", "channels", o_config.inChannels)) {
        return false;
    }

    if (!getValFromCfg(i_filename, "region", "classes", o_config.numClasses)) {
        return false;
    }

    if (!getValFromCfg(i_filename, "region", "num", o_config.numBoxes)) {
        return false;
    }

    getValsFromCfg(i_filename, "region", "anchors", o_config.anchors);
    if ((int)o_config.anchors.size() != o_config.numBoxes*2) {
        printf("ERROR: number of anchor points (=%d) must be equal to number of boxes (=%d)\n", 
            (int)o_config.anchors.size(), o_config.numBoxes);
        return false;
    }

    o_config.outChannels = o_config.numBoxes*(5+o_config.numClasses); // 5 = box coords+confidence
    o_config.outWidth  = i_outputWidth;
    o_config.outHeight = i_outputWidth;
    o_config.nms       = i_nms;

    return true;
}

bool readNamesFile(const std::string& i_filename, std::vector<std::string>& o_classNames)
{
	o_classNames.clear();

	std::ifstream is(i_filename);
	if (!is.is_open()) {
		printf("Could not open file %s\n", i_filename.c_str());
		return false;
	}

	std::string line;
	while (std::getline(is, line)) {
		line = trim(line);
		o_classNames.push_back(line);
	}

	return true;
}