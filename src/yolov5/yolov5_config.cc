#include "include/yolov5/yolov5_config.h"


namespace yolov5 {

std::shared_ptr<Yolov5Config> Yolov5Config::yolov5Config = nullptr;

void Yolov5Config::setParameterFile(const std::string &filename) {
    if (yolov5Config == nullptr) {
        yolov5Config = std::shared_ptr<Yolov5Config>(new Yolov5Config);
    }
    yolov5Config->file = cv::FileStorage(filename, cv::FileStorage::READ);
    if (!yolov5Config->file.isOpened()) {
        std::cerr << "YOLOv5 parameter file <" << filename << "> does not exist." << std::endl;
        yolov5Config->file.release();
    }
}


void Yolov5Config::setRootPath(const std::string &path) {
    if (path.length() == 0) {
        std::cerr << "root path is empty." << std::endl;
        exit(1);
    }
    yolov5Config->rootPath = path;
}


Yolov5Config::~Yolov5Config() {
    if (yolov5Config->file.isOpened()) {
        yolov5Config->file.release();
    }
}

}
