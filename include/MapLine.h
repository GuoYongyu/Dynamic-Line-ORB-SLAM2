#ifndef MAPLINE_H
#define MAPLINE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/core/core.hpp>

#include <mutex>
#include <eigen3/Eigen/Core>
#include <map>


using namespace cv::line_descriptor;
using namespace Eigen;


namespace ORB_SLAM2 {

class KeyFrame;
class Map;
class Frame;


typedef Matrix<double, 6, 1> Vector6d;


class MapLine {
public:
    // Similar to PL-SLAM
    // MapLine(const int& idx_, const Vector6d& line3D_, const cv::Mat& desc_, 
    //         const int& kfObs_, const Vector3d& obs_, const Vector3d& dir_,
    //         const Vector4d& pts_, const double& sigma2_ = 1.f);
    // ~MapLine() {};

    // void addMapLineObservation(const cv::Mat& desc_, const int& kfObs_, const Vector3d& obs_, const Vector3d& dir_,
    //                            const Vector4d& pts_, const double& sigma2_ = 1.f);
    
    // vector<Vector3d> mvObsList;  // 每个观察点的坐标，2D线方程，用sqrt(lx^2 + ly^2)归一化
    // vector<Vector4d> mvPtsList;  // 线特征两个端点的坐标

    // vector<int>    mvKfObsList;  // 观测到线特征所在关键帧的ID
    // vector<double> mvSigmaList;  // 每个观测值的sigma尺度

    // Similar to ORB-SLAM2
    // MapLine created by key frame
    MapLine(const Vector6d& pos, KeyFrame* pRefKF, Map* pMap, const string& sLabel);
    MapLine(const Vector6d& pos, KeyFrame* pRefKF, Map* pMap);
    // MapLine created by usual frame
    MapLine(const Vector6d& pos, Frame* pFrame, Map* pMap, const int& idxF, const string& sLabel);

    void SetWorldPos(const Vector6d& pos);
    Vector6d GetWorldPos();

    string GetSemanticLabel();

    Vector3d GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    map<KeyFrame*, size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF, size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapLine* pML);
    MapLine* GetReplaced();

    void IncreaseVisible(const int& n = 1);
    void IncreaseFound(const int& n = 1);
    float GetFoundRatio();
    inline int GetFound() {
        return mnFound;
    }

    // Similart to ORB-SLAM2, compute the distinctive descriptors of line features
    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    // Compute average direction of line features
    void UpdateAverageDir();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float& currentDist, const float& logScaleFactor);

public:
    long unsigned int mnId;  // Global ID for MapLine
    static long unsigned int nNextId;
    const long int mnFirstKfId;  // ID of key frame which creats this MapLine
    const long int mnFirstFrame;  // ID of frame which creats this MapLine
    int nObs;

    // Variables used by the tracking
    float mTrackProjX1;
    float mTrackProjY1;
    float mTrackProjX2;
    float mTrackProjY2;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    // TrackLocalMap - SearchByProjection中决定是否对该特征线进行投影的变量
    // mbTrackInView == false的点有几种：
    // a. 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b. 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    // c. 不在当前相机视野中的点（即未通过isInFrustum判断）
    bool mbTrackInView;
    // TrackLocalMap - UpdateLocalLines中防止将MapLines重复添加至mvpLocalMapLines的标记
    long unsigned int mnTrackReferenceForFrame;
    // TrackLocalMap - SearchLocalLines中决定是否进行isInFrustum判断的变量
    // mnLastFrameSeen == mCurrentFrame.mnId的line有集中：
    // a. 已经和当前帧经过匹配（TrackReferenceKeyFrame, TrackWithMotionModel)，但在优化过程中认为是外点
    // b. 已经和当前帧经过匹配且为内点，这类line也不需要再进行投影
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopLineForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

public:
    // Position in world coordinate
    Vector6d mWorldPos;
    Vector3d mStart3D;
    Vector3d mEnd3D;

    string msSemanticLabel;

    // KeyFrame that observs the line and associated index in key frame
    map<KeyFrame*, size_t> mObservations;

    // In MapPoint, this is average observation direction of point
    // In MapLine, this is average observation direction of line segment
    Vector3d mNormalVector;

    // The best descriptor computed by ComputeDistinctiveDescriptors()
    cv::Mat mLDescriptor;

    // Reference key frame
    KeyFrame* mpRefKF;

    // Set of line feature's descriptors
    vector<cv::Mat> mvDescList;
    // Direction each observated line segment
    vector<Vector3d> mvDirList;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag, do not currently erase MapLine from memory
    bool mbBad;
    MapLine* mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map* mpMap;
    mutex mMutexPos;
    mutex mMutexFeatures;

};


}


#endif // MAPLINE_H
