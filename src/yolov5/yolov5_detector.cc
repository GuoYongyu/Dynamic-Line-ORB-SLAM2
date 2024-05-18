#include "include/yolov5/yolov5_detector.h"


template<typename T>
T maxOf4Vals(const T &a, const T &b,
             const T &c, const T &d) {
    T ret = T(0);
    ret = ret > a ? ret : a;
    ret = ret > b ? ret : b;
    ret = ret > c ? ret : c;
    ret = ret > d ? ret : d;
    return ret;
}


template<typename T>
T minVal(const T &a, const T &b) {
    return a < b ? a : b;
}


template<typename T>
T maxVal(const T &a, const T &b) {
    return a > b ? a : b;
}


namespace yolov5 {


Yolov5Type parseYoloModelType(const std::string &type) {
    if (type == "YOLOv5L") {
        return YOLOv5L;
    } else if (type == "YOLOv5M") {
        return YOLOv5M;
    } else if (type == "YOLOv5N") {
        return YOLOv5N;
    } else if (type == "YOLOv5S") {
        return YOLOv5S;
    } else if (type == "YOLOv5X") {
        return YOLOv5X;
    } else {
        std::cerr << "Wrong YOLOv5 model type: " << type;
        exit(1);
    }
}


bool maskPredicate(const cv::KeyPoint &keyPoint, const cv::Mat &mask) {
    return mask.at<uchar>((int)(keyPoint.pt.y + 0.5f), (int)(keyPoint.pt.x + 0.5f)) == 0;
}


std::set<std::string> Yolov5Detector::dynamicObjects = {
    "person", "bicycle", "car", "motorbike", "bus", "truck"
};
std::set<std::string> Yolov5Detector::planarObjects = {
    "book", "keyboard", "laptop", "tvmonitor"
};
std::map<std::string, std::vector<float>> Yolov5Detector::planarColorMap = {
    {"book",      {0.f, 0.5f, 0.f}},
    {"keyboard",  {0.f, 1.f, 1.f}},
    {"laptop",    {0.6f, 0.f, 0.5f}},
    {"tvmonitor", {1.f, 1.f, 0.f}},
    {"common",    {0.f, 0.f, 1.f}}
};


Yolov5Detector::Yolov5Detector(const Yolov5Type &yolov5Type) {
    this->yolov5Type = yolov5Type;

    labelsFile = Yolov5Config::yolov5Config->rootPath + Yolov5Config::get<std::string>("coco_names_file");
    std::ifstream labelsFilePtr(labelsFile.c_str());
    std::string line;
    while (std::getline(labelsFilePtr, line)) {
        labels.push_back(line);
    }

    objectThickness = Yolov5Config::get<float>("object_thickness");
    farAwayThresh = Yolov5Config::get<float>("far_away_thresh");
    removalBlockThresh = Yolov5Config::get<double>("removal_block_thresh");

    // imageWidth = Yolov5Config::get<int>("image_width");
    // imageHeight = Yolov5Config::get<int>("image_height");

    numBoxSample = Yolov5Config::get<int>("num_grid_sample");
    minDepthChangeRate = Yolov5Config::get<double>("min_depth_change_rate");
    dynamicBoxExpansionRate = Yolov5Config::get<double>("dynamic_box_expansion_rate");
    planarBoxExpansionRate = Yolov5Config::get<double>("planar_box_expansion_rate");
    lineIntersectionRate = Yolov5Config::get<double>("line_intersection_rate");

    initYoloSetting();

    std::cout << std::endl << "Loading YOLOv5 model..." << std::endl;
    net = cv::dnn::readNetFromONNX(Yolov5Config::yolov5Config->rootPath + modelFile);
    std::cout << "Finished load YOLOv5 model." << std::endl;

    useSemanticMask = Yolov5Config::get<int>("use_semantic_mask");
    if (0 == useSemanticMask) {
        std::cout << "Semantic Mask Extracting is closed." << std::endl;
    } else {
        std::cout << "Semantic Mask Extracting is used" << std::endl;
    }

    useCUDA = Yolov5Config::get<int>("use_cuda");
    if (useCUDA) {
        std::cout << std::endl << "Use GPU to process YOLOv5." << std::endl;
        try {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } catch (cv::Exception) {
            std::cout << "Meet error, use CPU to process YOLOv5." << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    } else {
        std::cout << std::endl << "Use CPU to process YOLOv5." << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}


bool Yolov5Detector::isUsingSemanticMask() {
    if (0 == useSemanticMask) {
        return false;
    } else {
        return true;
    }
}


void Yolov5Detector::setImageSize(const int &width, const int &height) {
    imageWidth = width;
    imageHeight = height;
}


void Yolov5Detector::setDepthFactor(const double &factor) {
    depthFactor = factor;
}


void Yolov5Detector::initYoloSetting() {
    std::string name;
    switch (yolov5Type) {
        case YOLOv5L:
            name = "yolov5l";
            break;
        case YOLOv5M:
            name = "yolov5m";
            break;
        case YOLOv5N:
            name = "yolov5n";
            break;
        case YOLOv5S:
            name = "yolov5s";
            break;
        case YOLOv5X:
            name = "yolov5x";
            break;
        default:
            std::cerr << "Unknown YOLOv5 model." << std::endl;
            exit(1);
    }

    modelFile = Yolov5Config::get<std::string>(name + "_model_file");
    modelName = Yolov5Config::get<std::string>(name + "_net_name");
    yolov5Setting.confThresh = Yolov5Config::get<float>("yolov5_confidence_thresh");
    yolov5Setting.nmsThresh = Yolov5Config::get<float>("yolov5_nms_thresh");
    yolov5Setting.boxThresh = Yolov5Config::get<float>("yolov5_box_thresh");
    yolov5Setting.inputWidth = Yolov5Config::get<int>("yolov5_input_width");
    yolov5Setting.inputHeight = Yolov5Config::get<int>("yolov5_input_height");

    std::cout << std::endl << "Using " << modelName << " model." << std::endl;
}


void Yolov5Detector::warmUpModel() {
    // warm up the YOLO to decrease the time consumption of the first frame
    std::cout << std::endl << "Warming up YOLOv5 model..." << std::endl;
    cv::Mat img = cv::Mat::zeros(Yolov5Detector::imageHeight, Yolov5Detector::imageWidth, CV_8UC3);
    getNetOutputs(img);
    std::cout << "Finished warm up YOLOv5 model." << std::endl;
}


void Yolov5Detector::getNetOutputs(const cv::Mat &img) {
    auto t1 = std::chrono::steady_clock::now();

    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1 / 255.0,
                           cv::Size(yolov5Setting.inputWidth, yolov5Setting.inputHeight),
                           cv::Scalar(0, 0, 0), true, false);
//    cv::dnn::blobFromImage(img, blob, 1 / 255.0,
//                           cv::Size(yolov5Setting.inputWidth, yolov5Setting.inputHeight),
//                           cv::Scalar(104, 117, 123), true, false);
//    cv::dnn::blobFromImage(img, blob, 1 / 255.0,
//                           cv::Size(yolov5Setting.inputWidth, yolov5Setting.inputHeight),
//                           cv::Scalar(114, 114, 114), true, false);
    net.setInput(blob);
    // for some OpenCV versions, getUnconnectedOutLayersNames() may not get "output" layer
    net.forward(netOutputs, "output");
    // net.forward(netOutputs, net.getUnconnectedOutLayersNames());

    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "(YOLOv5) Get net outputs time: " << double(duration.count()) / 1000.0 << "ms" << std::endl;
}


cv::Mat Yolov5Detector::filterFarPoints(cv::Mat &depth) const {
    auto t1 = std::chrono::steady_clock::now();

    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8U);
    for (int i = 0; i < depth.rows; i++) {
        for (int j = 0; j < depth.cols; j++) {
            if (double(depth.at<ushort>(i, j)) > farAwayThresh * depthFactor) {
                depth.at<ushort>(i, j) = 0;
                mask.at<uchar>(i, j) = 0xff;
            }
        }
    }

    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "(YOLOv5) Filter far away points time: " << double(duration.count()) / 1000.0 << "ms" << std::endl;
    return mask;
}


void Yolov5Detector::getObjectBoxes(const cv::Mat &color, std::vector<ObjectBox> &dynamicBoxes,
                                    std::vector<ObjectBox> &planarBoxes) {
    getNetOutputs(color);

    auto t1 = std::chrono::steady_clock::now();

    std::vector<int> labelIDs;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float ratioH = float(imageHeight) / float(yolov5Setting.inputHeight);
    float ratioW = float(imageWidth) / float(yolov5Setting.inputWidth);
    // the width of net output is 5 over the number of the class IDs
    int netWidth = netOutputs[0].size[2];
    auto pData = (float*)netOutputs[0].data;
    int netHeight = netOutputs[0].size[1];
    for (int r = 0; r < netHeight; ++r, pData += netWidth) {
        float boxScore = pData[4];  // the probability of whether there is object in current box
        if (boxScore > yolov5Setting.boxThresh) {
            cv::Mat scores(1, int(labels.size()), CV_32FC1, pData + 5);
            cv::Point labelIdPoint;
            double maxLabelScore;
            cv::minMaxLoc(scores, nullptr, &maxLabelScore, nullptr, &labelIdPoint);
            maxLabelScore *= boxScore;

            if (maxLabelScore > yolov5Setting.confThresh) {
                auto x = pData[0];
                auto y = pData[1];
                auto w = pData[2];
                auto h = pData[3];
                int left = maxVal(int(std::round(x - 0.5 * w) * ratioW), 0);
                int top = maxVal(int(std::round(y - 0.5 * h) * ratioH), 0);
                labelIDs.push_back(labelIdPoint.x);
                confidences.push_back(float(maxLabelScore));
                boxes.push_back(cv::Rect(left, top, int(std::round(ratioW * w)), int(std::round(ratioH * h))));
            }
        }
    }

    // perform non maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, yolov5Setting.confThresh, yolov5Setting.nmsThresh, indices);

    for (auto& idx: indices) {
        if (dynamicObjects.find(labels[labelIDs[idx]]) != dynamicObjects.end()) {
            dynamicBoxes.emplace_back(boxes[idx], labels[labelIDs[idx]]);
        } else if (planarObjects.find(labels[labelIDs[idx]]) != planarObjects.end()) {
            planarBoxes.emplace_back(boxes[idx], labels[labelIDs[idx]]);
        }
    }
    std::cout << "(YOLOv5) Dynamic boxes num: " << dynamicBoxes.size() << std::endl;
    std::cout << "(YOLOv5) Planar boxes num: " << planarBoxes.size() << std::endl;

    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "(YOLOv5) Parsing boxes time: " << double(duration.count()) / 1000.0 << "ms" << std::endl;
}


cv::Mat Yolov5Detector::getImageMask(const std::vector<ObjectBox> &dynamicBoxes, const cv::Mat &depth,
                                     const bool &sampledCenter, const bool &filterZeroDepth, const bool &expandBox) const {
    if (0 == useSemanticMask) {
        return cv::Mat::zeros(depth.size(), CV_8U);
    }

    auto t1 = std::chrono::steady_clock::now();

    // get mask
    cv::Mat finalMask = cv::Mat::zeros(depth.size(), CV_8U);
    for (auto& box: dynamicBoxes) {
        cv::Mat boxMask = getMaskOfBox(
                box.box, depth, depthFactor,
                objectThickness, dynamicBoxExpansionRate,
                numBoxSample, minDepthChangeRate,
                sampledCenter, filterZeroDepth, expandBox);
        cv::bitwise_or(finalMask, boxMask, finalMask);
    }

    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "(YOLOv5) Get image mask time: " << double(duration.count()) / 1000.0 << "ms" << std::endl;
    return finalMask;
}

}
