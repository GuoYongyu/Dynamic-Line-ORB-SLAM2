#include "include/yolo_utils/bbox.h"

#include <opencv2/line_descriptor/descriptor.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>


std::ostream& operator<<(std::ostream &out, const ObjectBox &box) {
    auto x = box.box.x, y = box.box.y, w = box.box.width, h = box.box.height;
    std::cout << "[(" << x << ", " << y << "), (" << x + w << ", " << y + h << ")]";
    return out;
}


cv::Rect getExpandedBox(const cv::Mat &image, const cv::Rect &box, const double &boxExpansionRate) {
    // expand the bounding box to cover more image
    int left = int(box.x - boxExpansionRate / 2.0 * box.width);
    int top = int(box.y - boxExpansionRate / 2.0 * box.height);
    // int left = box.x, top = box.y;
    int w = int((1.0 + boxExpansionRate) * double(box.width));
    int h = int((1.0 + boxExpansionRate) * double(box.height));

    left = std::max(0, std::min(left, image.cols - 1));
    top = std::max(0, std::min(top, image.rows - 1));
    w = std::min(w, image.cols - left);
    h = std::min(h, image.rows - top);
    auto new_box = cv::Rect(left, top, w, h);
    return new_box;
}


cv::Rect getExpandedBox(const int &nCol, const int &nRow, const cv::Rect& box, const double& boxExpansionRate) {
    // expand the bounding box to cover more image
    int left = int(box.x - boxExpansionRate / 2.0 * box.width);
    int top = int(box.y - boxExpansionRate / 2.0 * box.height);
    // int left = box.x, top = box.y;
    int w = int((1.0 + boxExpansionRate) * double(box.width));
    int h = int((1.0 + boxExpansionRate) * double(box.height));

    left = std::max(0, std::min(left, nCol - 1));
    top = std::max(0, std::min(top, nRow - 1));
    w = std::min(w, nCol - left);
    h = std::min(h, nRow - top);
    auto new_box = cv::Rect(left, top, w, h);
    return new_box;
}


ushort getBoxSampledDepth(const cv::Rect &box, const cv::Mat &depth,
                          const int &numBoxSample, const double &minDepthChangeRate) {
    std::vector<ushort> sampledDepth;
    auto numDeparts = numBoxSample + 1;
    for (int i = 1; i <= numBoxSample; i++) {
        for (int j = 1; j <= numBoxSample; j++) {
            auto d = depth.at<ushort>(box.y + i * box.height / numDeparts, box.x + j * box.width / numDeparts);
            sampledDepth.push_back(d);
        }
    }

//    // sort depths to ascend mode
//    std::sort(sampledDepth.begin(), sampledDepth.end());
//    // the depth of object is most impossibly at the position of first 1/3
//    auto d_c = sampledDepth[sampledDepth.size() / 3];
//    // counter the depth value that is closed to the d_c
//    uint k = 1, depthSum = 0;
//    for (auto& d: sampledDepth) {
//        if (d >= d_c && double(d - d_c) / d_c < minDepthChangeRate ||
//            d < d_c && double(d_c - d) / d_c < minDepthChangeRate) {
//            depthSum += d;
//            k++;
//        }
//    }
//    auto ret = ushort(depthSum / k);

    auto ret = getRansacDepth(sampledDepth);
    std::cout << "(YOLOv5) Get sampled depth value: " << ret << ", box " << box << std::endl;
    return ret;
}


cv::Mat getMaskOfBox(const cv::Rect &box, const cv::Mat &depth, const double &depthScale,
                     const double &objectThickness, const double &boxExpansionRate,
                     const int &numBoxSample, const double &minDepthChangeRate,
                     const bool &sampledCenter, const bool &filterZeroDepth, const bool &expandBox) {
    auto d_tl = depth.at<ushort>(box.y, box.x);
    auto d_tr = depth.at<ushort>(box.y, box.x + box.width);
    auto d_bl = depth.at<ushort>(box.y + box.height, box.x);
    auto d_br = depth.at<ushort>(box.y + box.height, box.x + box.width);

    auto d_max = std::max({d_tl, d_tr, d_bl, d_br});

    auto d_c = depth.at<ushort>(box.y + box.height / 2, box.x + box.width / 2);
    if (sampledCenter) {
        d_c = std::min(d_c, getBoxSampledDepth(box, depth, numBoxSample, minDepthChangeRate));
    }
    std::cout << "(YOLOv5) Actually used object depth value: " << d_c << ", box " << box << std::endl;

    ushort d_thresh;
    if (d_c > 0 && double(d_max - d_c) >= objectThickness * depthScale) {
        d_thresh = (d_max + d_c) / 2;
    } else if (d_c > 0 && double(d_max - d_c) < objectThickness * depthScale) {
        d_thresh = d_c + ushort(objectThickness * depthScale);
    } else if (d_max > 0 && d_c == 0) {
        d_thresh = d_max;
    } else {
        d_thresh = 0xffff;
    }

    cv::Rect newBox;
    if (expandBox) {
        newBox = getExpandedBox(depth, box, boxExpansionRate);
    } else {
        newBox = box;
    }
    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8U);
    mask(newBox) = depth(newBox) < d_thresh;
    if (!filterZeroDepth) {
        cv::Mat m = depth(newBox) == 0;
        cv::bitwise_or(m, mask(newBox), mask(newBox));
        // cv::bitwise_xor(m, mask(newBox), mask(newBox));
    }

    // normal programs
//    for (int i = newBox.y; i <= newBox.y + newBox.height; ++i) {
//        for (int j = newBox.x; j <= newBox.x + newBox.width; ++j) {
//            auto d = depth.at<ushort>(i, j);
//
//            if (!filterZeroDepth && d == 0) {
//                continue;
//            }
//
//            if (d < d_thresh) {
//                mask.at<uchar>(i, j) = 0xff;
//            }
//        }
//    }

    return mask;
}


// Determine whether two line segments intersect
Eigen::Vector2f lineIntersection(
        const Eigen::Vector2f &p1, const Eigen::Vector2f &q1, 
        const Eigen::Vector2f &p2, const Eigen::Vector2f &q2) {
    double a = 0, b = 0, c = 0;
    if (p1(0) == q1(0)) {
        c = p1(0);
    } else {
        double k = (q1(1) - p1(1)) / (q1(0) - p1(0));
        double h = p1(1) - k * p1(0);
        a = -k; b = 1.0; c = -h;
    }

    double d1 = signedDistance(a, b, c, p2);
    double d2 = signedDistance(a, b, c, q2);
    if (d1 * d2 <= 0) {
        d1 = abs(d1);
        d2 = abs(d2);
        double k = d1 / (d1 + d2);
        if (d1 + d2 != 0 && (p1(0) + k * (q1(0) - p1(0)) != p1(0) || p1(1) + k * (q1(1) - p1(1)) != p1(1))) {
            return Eigen::Vector2f(p1(0) + k * (q1(0) - p1(0)), p1(1) + k * (q1(1) - p1(1)));
        } else {
            return Eigen::Vector2f(-1, -1);
        }
    } else {
        return Eigen::Vector2f(-1, -1);
    }

    return Eigen::Vector2f(-1, -1);
}


bool isLineInRect(const Eigen::Vector2f &p, const Eigen::Vector2f *rect) {
    if ((p(0) > rect[0](0) && p(0) < rect[4](0)) &&
        (p(1) > rect[0](1) && p(0) < rect[4](1))) {
        return true;
    }
    return false;
}


// Judge whether the line segment and the rectangle intersect
// Calculate the length proportion of the intersection
double getIntersectionRatio(const cv::Rect &rect, const Eigen::Vector2f &lineStart, const Eigen::Vector2f &lineEnd) {
    Eigen::Vector2f rectPoints[4] = {{rect.x, rect.y}, {rect.x + rect.width, rect.y}, {rect.x, rect.y + rect.height}, {rect.x + rect.width, rect.y + rect.height}};
    
    bool sInRect = isLineInRect(lineStart, rectPoints);
    bool eInRect = isLineInRect(lineEnd, rectPoints);

    if (sInRect && eInRect) {
        return 1.0;
    }

    double totalLength = 0;
    double intersectLength = 0;

    std::vector<Eigen::Vector2f> ip(4);
    ip[0] = lineIntersection(lineStart, lineEnd, rectPoints[0], rectPoints[1]);
    ip[1] = lineIntersection(lineStart, lineEnd, rectPoints[0], rectPoints[2]);
    ip[2] = lineIntersection(lineStart, lineEnd, rectPoints[1], rectPoints[3]);
    ip[3] = lineIntersection(lineStart, lineEnd, rectPoints[2], rectPoints[3]);

    int count = 0, idx1 = -1, idx2 = -1;
    for (int i = 0; i < 4; i++) {
        if (ip[i](0) >= 0 && ip[i](1) >= 0) {
            count++;
            if (idx1 >= 0) {
                idx2 = i;
            } else {
                idx1 = i;
            }
        }
    }

    if (count == 2) {
        intersectLength = distance(ip[idx1], ip[idx2]);
    } else if (count == 1) {
        if (sInRect) {
            intersectLength = distance(lineStart, ip[idx1]);
        } else {
            intersectLength = distance(lineEnd, ip[idx1]);
        }
    } else {
        return 0;
    }

    return intersectLength / totalLength;
}


double distance(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2) {
    return sqrt(pow(p2(0) - p1(0), 2) + pow(p2(1) - p1(1), 2));
}


double signedDistance(const double& a, const double& b, const double& c, const Eigen::Vector2f &p) {
    double d = abs((a * p(0) + b * p(1) + c) / (sqrt(a * a + b * b)));
    if (b == 0) {
        if (p(0) <= c) {
            return -d;
        } else {
            return d;
        }
    } else {
        if (p(1) >= ((-a / b) * p(0) - c / b)) {
            return d;
        } else {
            return -d;
        }
    }
}


std::string getLineLabelInBox(
        const int& width, const int& height, const std::vector<ObjectBox> &boxes, 
        const cv::line_descriptor::KeyLine &line, 
        const double& expansionRate, const double& intersectionRate
    ) {
    for (auto& box: boxes) {
        cv::Rect newBox = getExpandedBox(width, height, box.box, expansionRate);
        double ratio = getIntersectionRatio(
                newBox, 
                Eigen::Vector2f(line.startPointX, line.startPointY), 
                Eigen::Vector2f(line.endPointX, line.endPointY)
        );
        if (ratio >= intersectionRate) {
            return box.label;
        }
    }
    return "common";
}


std::string getLineLabelInBox(
        const int& width, const int& height, const std::vector<ObjectBox>& boxes, 
        const Eigen::Vector2f& start, const Eigen::Vector2f& end,
        const double& expansionRate, const double& intersectionRate
    ) {
    for (auto& box: boxes) {
        cv::Rect newBox = getExpandedBox(width, height, box.box, expansionRate);
        double ratio = getIntersectionRatio(newBox, start, end);
        if (ratio >= intersectionRate) {
            return box.label;
        }
    }
    return "common";
}
