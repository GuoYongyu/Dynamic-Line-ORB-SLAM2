//
// Created by Star Rain on 2023/5/16.
//

#ifndef SEMANTICMASK_YOLOV5_CONFIG_H
#define SEMANTICMASK_YOLOV5_CONFIG_H

#include "yolov5_include.h"


namespace yolov5 {

class Yolov5Config {
private:
    cv::FileStorage file;

    Yolov5Config() = default;

public:
    std::string rootPath;

    static std::shared_ptr<Yolov5Config> yolov5Config;

    ~Yolov5Config();

    static void setParameterFile(const std::string& filename);
    static void setRootPath(const std::string& path);

    template<class T>
    static T get(const std::string& key) {
        return T(Yolov5Config::yolov5Config->file[key]);
    }
};

enum Yolov5Type {
    YOLOv5L,
    YOLOv5M,
    YOLOv5N,
    YOLOv5S,
    YOLOv5X
};

class Yolov5Setting {
public:
    float confThresh;
    float nmsThresh;
    float boxThresh;
    int inputWidth;
    int inputHeight;
    std::string weightFile;
    std::string configFIle;
    std::string modelName;
};

}


#endif //SEMANTICMASK_YOLOV5_CONFIG_H
