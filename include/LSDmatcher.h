#ifndef ORB_SLAM2_LSDMATCHER_H
#define ORB_SLAM2_LSDMATCHER_H

// #include <line_descriptor/descriptor_custom.hpp>
// #include <line_descriptor_custom.hpp>

#include "MapLine.h"
#include "KeyFrame.h"
#include "Frame.h"

using namespace cv::line_descriptor;
using namespace cv;


namespace ORB_SLAM2 {

class LSDmatcher {
public:
    LSDmatcher(float nnratio = 0.6, bool checkOri = true);

    // Tracking line in last frame by projection
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame);

    // Tracking local MapLine by projection
    int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th=3);

    int SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, vector<pair<int,int>> &LineMatches);

    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, vector<pair<size_t, size_t>> &vMatchedPairs);

    // Project MapLines into KeyFrame and search for duplicated MapLines
    int Fuse(KeyFrame* pKF, const vector<MapLine *> &vpMapLines);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    float RadiusByViewingCos(const float &viewCos);

    float mfNNratio;
    bool mbCheckOrientation;
};

}


#endif //ORB_SLAM2_LSDMATCHER_H
