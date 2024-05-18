#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/io/pcd_io.h>

#include "Converter.h"
#include "PointCloudMap.h"
#include "KeyFrame.h"

#include <boost/make_shared.hpp>


PointCloudMap::PointCloudMap(double _resolution): resolution(_resolution) {
    voxelFilter.setLeafSize(resolution, resolution, resolution);
    globalPointCloud = boost::make_shared<PointCloud>();

    viewerThread = make_shared<thread>(bind(&PointCloudMap::viewer, this));
}


void PointCloudMap::shutdown() {
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyframeUpdated.notify_one();
    }
    viewerThread->join();
}


void PointCloudMap::insertKeyFrame(
        KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, const cv::Mat& mask, 
        const bool& useSemanticMask, const size_t& numDynamicObjects
    ) {
    std::cout << std::endl << "(PointCloud) Insert one keyframe, frame id = " << kf->mnFrameId << std::endl;
    cv::Mat locations;
    cv::findNonZero(mask, locations);
    if (locations.rows == 0 && useSemanticMask && numDynamicObjects != 0) {
        // If mask is wrong, then this key frame will be thrown away
        std::cout << "(PointCloud) All value in mask is zero, frame id = " << kf->mnFrameId << std::endl;
        return;
    }
    unique_lock<mutex> lck(keyframeMutex);
    keyFrames.push_back(kf);
    colorImgs.push_back(color.clone());
    depthImgs.push_back(depth.clone());
    maskImgs.push_back(mask.clone());
    keyframeUpdated.notify_one();
}


pcl::PointCloud<PointCloudMap::PointRGB>::Ptr PointCloudMap::generatePointCloud(
        KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth, const cv::Mat& mask
    ) {
    PointCloud::Ptr tmp(new PointCloud());
    for (int i = 0; i < depth.rows; i += 3) {
        for (int j = 0; j < depth.cols; j += 3) {
            float d = depth.ptr<float>(i)[j];
            uchar m = mask.ptr<uchar>(i)[j];
            if ((d < 0.01 || d > 10) || m > 0) {
                continue;
            }
            
            PointRGB p;
            p.z = d;
            p.x = (j - kf->cx) * p.z / kf->fx;
            p.y = (i - kf->cy) * p.z / kf->fy;
            p.b = color.ptr<uchar>(i)[j * 3];
            p.g = color.ptr<uchar>(i)[j * 3 + 1];
            p.r = color.ptr<uchar>(i)[j * 3 + 2];

            tmp->points.push_back(p);
        }
    }
    std::cout << std::endl;

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(kf->GetPose());
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud(*tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    std::cout << std::endl << "(PointCloud) Generated point cloud for keyframe(frame id = " << kf->mnFrameId << "), size = " << cloud->points.size() << std::endl;
    return cloud;
}


void PointCloudMap::viewer() {
    std::cout << std::endl << "(PointCloud) Listening key frame..." << std::endl;
    // pcl::visualization::CloudViewer viewer("viewer");
    while (true) {
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag) {
                std::cout << std::endl << "-------" << std::endl;
                std::cout << "(PointCloud) Received " << keyFrames.size() << " key frames." << std::endl;
                std::cout << "(PointCloud) Saving global point cloud to Output/GlobalPointCloud.pcd ..." << std::endl << std::endl;
                pcl::io::savePCDFile("./Output/GlobalPointCloud.pcd", *globalPointCloud);
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated(keyframeUpdatedMutex);
            keyframeUpdated.wait(lck_keyframeUpdated);
        }

        // keyframe is updated
        size_t N = 0;
        {
            unique_lock<mutex> lck(keyframeMutex);
            N = keyFrames.size();
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        for (size_t i = lastKeyframeSize; i < N; i++) {
            PointCloud::Ptr cloud = generatePointCloud(keyFrames[i], colorImgs[i], depthImgs[i], maskImgs[i]);
            *globalPointCloud += *cloud;
        }
        PointCloud::Ptr tmp(new PointCloud());
        if (globalPointCloud->size() > uint(double(INT_MAX) * resolution)) {
            resolution += resolutionInrement;
            voxelFilter.setLeafSize(resolution, resolution, resolution);
            std::cout << std::endl << "(PointCloud) Point cloud is too large, update resolution = " << resolution << std::endl;
        }
        voxelFilter.setInputCloud(globalPointCloud);
        voxelFilter.filter(*tmp);
        globalPointCloud->swap(*tmp);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        cout << "(PointCloud) Average point cloud updating: " << double(duration.count()) / 1000.0 / N << "ms" << endl;

        // viewer.showCloud(globalPointCloud);
        std::cout << std::endl << "(PointCloud) Update global point cloud, size = " << globalPointCloud->size() << std::endl;
        lastKeyframeSize = N;
    }
}
