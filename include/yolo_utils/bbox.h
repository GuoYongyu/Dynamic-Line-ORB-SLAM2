// Used to process bounding box created by YOLO

#ifndef SEMANTICMASK_BBOX_H
#define SEMANTICMASK_BBOX_H


#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "ransac.h"


class ObjectBox {
public:
    cv::Rect box;
    std::string label;

    ObjectBox(const cv::Rect& b, const std::string& l): box(b), label(l) {}
};

std::ostream& operator<<(std::ostream& out, const ObjectBox& box);


// expend existed bounding box to 1 + expansion_rate times bigger
cv::Rect getExpandedBox(const cv::Mat& image, const cv::Rect& box, const double& boxExpansionRate);
cv::Rect getExpandedBox(const int& nCol, const int& nRow, const cv::Rect& box, const double& boxExpansionRate);

// sample a depth value in depth image
ushort getBoxSampledDepth(const cv::Rect& box, const cv::Mat& depth,
                          const int& numBoxSample, const double& minDepthChangeRate);

// get mask of image with one bounding box
cv::Mat getMaskOfBox(const cv::Rect& box, const cv::Mat& depth, const double& depthScale,
                     const double& objectThickness, const double& boxExpansionRate,
                     const int& numBoxSample, const double& minDepthChangeRate,
                     const bool& sampledCenter, const bool& filterZeroDepth, const bool& expandBox);

// check wether line is in box, 80% of line is in box, return the label of box that line belongs to
std::string getLineLabelInBox(
    const int& width, const int& height, const std::vector<ObjectBox>& boxes, 
    const cv::line_descriptor::KeyLine& line, 
    const double& expansionRate = 0.0, const double& intersectionRate = 0.95);
std::string getLineLabelInBox(
    const int& width, const int& height, const std::vector<ObjectBox>& boxes, 
    const Eigen::Vector2f& start, const Eigen::Vector2f& end,
    const double& expansionRate = 0.0, const double& intersectionRate = 0.95);

// auxiliary functions
Eigen::Vector2f lineIntersection(
    const Eigen::Vector2f &p1, const Eigen::Vector2f &q1, 
    const Eigen::Vector2f &p2, const Eigen::Vector2f &q2);
bool isLineInRect(const Eigen::Vector2f &p, const Eigen::Vector2f *rect);
double distance(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2);
double signedDistance(const double& a, const double& b, const double& c, const Eigen::Vector2f &p);


#endif //SEMANTICMASK_BBOX_H
