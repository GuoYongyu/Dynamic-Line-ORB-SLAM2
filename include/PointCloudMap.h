#ifndef POINTCLOUDMAP_H
#define POINTCLOUDMAP_H


#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <condition_variable>

#include "System.h"

using namespace ORB_SLAM2;

class PointCloudMap {
public:
    typedef pcl::PointXYZRGB PointRGB;
    typedef pcl::PointCloud<PointRGB> PointCloud;

    PointCloudMap(double _resolution);

    void insertKeyFrame(
        KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, const cv::Mat& mask, 
        const bool& useSemanticMask, const size_t& numDynamicObjects
    );
    void shutdown();
    void viewer();

protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, const cv::Mat& mask);

    PointCloud::Ptr globalPointCloud;
    shared_ptr<thread> viewerThread;

    bool shutDownFlag = false;
    mutex shutDownMutex;

    condition_variable keyframeUpdated;
    mutex              keyframeUpdatedMutex;

    std::vector<KeyFrame*> keyFrames;
    std::vector<cv::Mat>   colorImgs;
    std::vector<cv::Mat>   depthImgs;
    std::vector<cv::Mat>   maskImgs;
    mutex                  keyframeMutex;
    uint16_t               lastKeyframeSize = 0;

    double resolution = 0.04;
    double resolutionInrement = 0.005;
    pcl::VoxelGrid<PointRGB> voxelFilter;
};


#endif // POINTCLOUDMAP_H
