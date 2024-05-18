//
// Created by Star Rain on 2023/5/16.
//

#ifndef SEMANTICMASK_YOLOV5_DETECTOR_H
#define SEMANTICMASK_YOLOV5_DETECTOR_H

#include "yolov5_config.h"
#include "yolov5_include.h"

// reference: https://github.com/UNeedCryDear/yolov5-opencv-dnn-cpp

namespace yolov5 {


class Yolov5Detector {
private:
    Yolov5Type yolov5Type;
    Yolov5Setting yolov5Setting;

    std::string labelsFile;
    std::string modelFile;
    std::string modelName;

    int useSemanticMask;

    int                  useCUDA;
    cv::dnn::Net         net;
    std::vector<cv::Mat> netOutputs;

    std::vector<std::string> labels;

    const std::vector<std::vector<float>> netAnchors = {
            {10.0, 13.0, 16.0, 30.0, 33.0, 23.0},
            {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},
            {116.0, 90.0, 156.0, 198.0, 373.0, 326.0}
    };
    const std::vector<float> netStride = {8.0, 16.0, 32.0};

    float  objectThickness;
    float  farAwayThresh;
    double removalBlockThresh;

    double depthFactor;

    int    imageWidth, imageHeight;
    // sample numBoxSample * numBoxSample depth points in bounding box
    int    numBoxSample;
    // min depth change rate in a bounding box when the depth point falls on the object
    double minDepthChangeRate;
    // the rate of bounding box expansion
    double dynamicBoxExpansionRate, planarBoxExpansionRate;
    // the rate of line intersecting with bounding box
    double lineIntersectionRate;

    void getNetOutputs(const cv::Mat& img);
    void initYoloSetting();

public:
    typedef std::shared_ptr<Yolov5Detector> Ptr;

    static std::set<std::string>    dynamicObjects;
    static std::set<std::string>    planarObjects;
    static std::map<std::string, std::vector<float>> planarColorMap;

    explicit Yolov5Detector(const Yolov5Type& yolov5Type);

    void setImageSize(const int& width, const int& height);
    void setDepthFactor(const double& factor);
    cv::Mat filterFarPoints(cv::Mat& depth) const;
    void warmUpModel();
    void getObjectBoxes(const cv::Mat& color, std::vector<ObjectBox>& dynamicBoxes, std::vector<ObjectBox>& planarBoxes);
    cv::Mat getImageMask(const std::vector<ObjectBox>& dynamicBoxes, const cv::Mat& depth,
                         const bool& sampledCenter, const bool& filterZeroDepth, const bool& expandBox) const;

    bool isUsingSemanticMask();

    double getPlanarBoxExpansionRate() {
        return this->planarBoxExpansionRate;
    }

    double getLineIntersectionRate() {
        return this->lineIntersectionRate;
    }

};

Yolov5Type parseYoloModelType(const std::string& type);

bool maskPredicate(const cv::KeyPoint& keyPoint, const cv::Mat& mask);

}


#endif //SEMANTICMASK_YOLOV5_DETECTOR_H
