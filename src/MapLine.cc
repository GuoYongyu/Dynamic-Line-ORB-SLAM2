#include "MapLine.h"

#include <mutex>
#include <map>

#include "LSDmatcher.h"


namespace ORB_SLAM2 {

mutex MapLine::mGlobalMutex;
long unsigned int MapLine::nNextId = 0;


MapLine::MapLine(const Vector6d& pos, KeyFrame* pRefKF, Map* pMap, const string& sLabel):
        mnFirstKfId(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
        mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
        mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
        mpReplaced(static_cast<MapLine*>(nullptr)), mpMap(pMap), msSemanticLabel(sLabel) {
    mWorldPos = pos;
    mNormalVector << 0, 0, 0;

    // MapLines can be created fram Tracking and LocalMapping, this mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}


MapLine::MapLine(const Vector6d &pos, KeyFrame *pRefKF, Map *pMap):
        mnFirstKfId(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
        mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
        mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
        mpReplaced(static_cast<MapLine*>(nullptr)), mpMap(pMap) {
    mWorldPos = pos;
    mNormalVector << 0, 0, 0;

    // Get semantic label of line
    cv::Mat start3Dw, end3Dw;
    start3Dw = (cv::Mat_<float>(3, 1) << float(pos(0)), float(pos(1)), float(pos(2)));
    end3Dw = (cv::Mat_<float>(3, 1) << float(pos(3)), float(pos(4)), float(pos(5)));
    cv::Mat Tcw = pRefKF->GetPose();
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    cv::Mat start3Dc = Rcw * start3Dw + tcw;
    cv::Mat end3Dc = Rcw * end3Dw + tcw;
    float xs = start3Dc.at<float>(0), ys = start3Dc.at<float>(1), zs = start3Dc.at<float>(2);
    float xe = end3Dc.at<float>(0), ye = end3Dc.at<float>(1), ze = end3Dc.at<float>(2);
    float us = xs * pRefKF->fx / zs + pRefKF->cx, vs = ys * pRefKF->fy / zs + pRefKF->cy;
    float ue = xe * pRefKF->fx / ze + pRefKF->cx, ve = ye * pRefKF->fy / ze + pRefKF->cy;
    msSemanticLabel = getLineLabelInBox(
        pRefKF->mWidth, pRefKF->mHeight, pRefKF->mvPlanarBoxes, 
        Vector2f(us, vs), Vector2f(ue, ve), 
        pRefKF->mPlanarBoxExpansionRate, pRefKF->mLineIntersectionRate
    );
    // cout << msSemanticLabel << endl;

    // MapLines can be created fram Tracking and LocalMapping, this mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}


MapLine::MapLine(const Vector6d& pos, Frame* pFrame, Map* pMap, const int& idxF, const string& sLabel):
        mnFirstKfId(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
        mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
        mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
        mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap), msSemanticLabel(sLabel) {
    mWorldPos = pos;
    cv::Mat Ow = pFrame->GetCameraCenter();  // Camera optical center coordinate
    Vector3d OW;
    OW << Ow.at<double>(0), Ow.at<double>(1), Ow.at<double>(2);
    mStart3D = pos.head(3);
    mEnd3D = pos.tail(3);
    Vector3d midPoint = 0.5 * (mStart3D + mEnd3D);
    mNormalVector = midPoint - OW;  // Vector from camera to midpoint of line in world coordinate
    mNormalVector.normalize();
    const float dist = mNormalVector.norm();  // Distance from camera to midpoint of line

    const int level = pFrame->mvKeyLinesUn[idxF].octave;
    const float levelScaleFactor = pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

    pFrame->mLdesc.row(idxF).copyTo(mLDescriptor);

    // MapLines can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexLineCreation);
    mnId = nNextId++;
}


void MapLine::SetWorldPos(const Vector6d &pos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = pos;
}


Vector6d MapLine::GetWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}


string MapLine::GetSemanticLabel() {
    unique_lock<mutex> lock(mMutexPos);
    return msSemanticLabel;
}


Vector3d MapLine::GetNormal() {
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector;
}


KeyFrame* MapLine::GetReferenceKeyFrame() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}


void MapLine::AddObservation(KeyFrame *pKF, size_t idx) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF)) {
        return;
    }
    // Save the key frame that observes the MapLine and corresponding index
    mObservations[pKF] = idx;
    nObs++;
}


void MapLine::EraseObservation(KeyFrame *pKF) {
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF)) {
            nObs -= 1;

            mObservations.erase(pKF);

            // If this key frame is reference frame, then reassigned RefFrame after erasing
            if (mpRefKF == pKF) {
                mpRefKF = mObservations.begin()->first;
            }

            // When the number of observation that camera observes the MapLine is less than 2
            // Then delete the line
            if (nObs <= 2) {
                bBad = true;
            }
        }
    }

    if (bBad) {
        // cout << "(MapLine) ID: " << mnId << ", bad flag" << endl;
        SetBadFlag();
    }
}


map<KeyFrame*, size_t> MapLine::GetObservations() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}


int MapLine::Observations() {
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}


int MapLine::GetIndexInKeyFrame(KeyFrame *pKF) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF)) {
        return mObservations[pKF];
    } else {
        return -1;
    }
}


bool MapLine::IsInKeyFrame(KeyFrame *pKF) {
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}


void MapLine::SetBadFlag() {
    // Notify the Frame that can observe the MapLine
    // The MapLine has been deleted
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        // Shallow copy
        obs = mObservations;
        mObservations.clear();
    }

    for (map<KeyFrame*, size_t>::iterator it = obs.begin(), end = obs.end();
         it != end; it++) {
        KeyFrame* pKF = it->first;
        pKF->EraseMapLineMatch(it->second);        
    }

    mpMap->EraseMapLine(this);
}


bool MapLine::isBad() {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}


void MapLine::Replace(MapLine *pML) {
    if (pML->mnId == this->mnId) {
        return;
    }

    int nVisible, nFound;
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
        nVisible = mnVisible;
        nFound = mnFound;
        mpReplaced = pML;
    }

    // Replace all key frames that can observe the MapLine
    for (map<KeyFrame*, size_t>::iterator it = obs.begin(), end = obs.end();
         it != end; it++) {
        KeyFrame* pKF = it->first;

        if (!pML->IsInKeyFrame(pKF)) {
            pKF->ReplaceMapLineMatch(it->second, pML);
            pML->AddObservation(pKF, it->second);
        } else {
            pKF->EraseMapLineMatch(it->second);
        }
    }

    pML->IncreaseFound(nFound);
    pML->IncreaseVisible(nVisible);
    pML->ComputeDistinctiveDescriptors();

    mpMap->EraseMapLine(this);
}


MapLine* MapLine::GetReplaced() {
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}


void MapLine::IncreaseVisible(const int &n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}


void MapLine::IncreaseFound(const int &n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}


float MapLine::GetFoundRatio() {
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound / mnVisible);
}


void MapLine::ComputeDistinctiveDescriptors() {
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*, size_t> observations;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if (mbBad) {
            return;
        }
        observations = mObservations;
    }

    if (observations.empty()) {
        return;
    }

    vDescriptors.reserve(observations.size());

    // Traverse all keyframes that observe 3D feature lines
    // Obtain LBD descriptors and insert them into vDescriptors
    for (map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end();
         mit!=mend; mit++) {
        KeyFrame* pKF = mit->first;

        if (!pKF->isBad()) {
            vDescriptors.push_back(pKF->mLineDescriptors.row(mit->second));
        }
    }

    if (vDescriptors.empty()) {
        return;
    }

    // Compute the distances between descriptors
    const size_t NL = vDescriptors.size();
    vector<vector<int>> distances;
    distances.resize(NL, vector<int>(NL, 0));
    for (size_t i = 0; i < NL; i++) {
        distances[i][i] = 0;
        for (size_t j = 0; j < NL; j++) {
            int distIJ = cv::norm(vDescriptors[i], vDescriptors[j], NORM_HAMMING);
            distances[i][j] = distIJ;
            distances[j][i] = distIJ;
        }
    }

    // Take the descriptor with least median distance to the rest
    int bestMedian = INT_MAX;
    int bestIdx = 0;
    for (size_t i = 0; i < NL; i++) {
        vector<int> vDists(distances[i].begin(), distances[i].end());
        sort(vDists.begin(), vDists.end());
        int median = vDists[int(0.5 * (NL - 1))];

        // Find the least median
        if (median < bestMedian) {
            bestMedian = median;
            bestIdx = i;
        }
    }
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mLDescriptor = vDescriptors[bestIdx].clone();
    }
}


cv::Mat MapLine::GetDescriptor() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mLDescriptor.clone();
}


void MapLine::UpdateAverageDir() {
    map<KeyFrame*, size_t> observations;
    KeyFrame* pRefKF;
    Vector6d pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if (mbBad) {
            return;
        }
        observations = mObservations;
        pRefKF = mpRefKF;
        pos = mWorldPos;
    }

    if (observations.empty()) {
        return;
    }

    Vector3d normal(0, 0, 0);
    int n = 0;
    for (map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); 
         mit != mend; mit++) {
        KeyFrame* pKF = mit->first;
        Mat Owi = pKF->GetCameraCenter();
        Vector3d OWi(Owi.at<float>(0), Owi.at<float>(1), Owi.at<float>(2));
        Vector3d middlePos = 0.5 * (mWorldPos.head(3) + mWorldPos.tail(3));
        Vector3d normali = middlePos - OWi;
        normal = normal + normali / normali.norm();
        n++;
    }

    cv::Mat SP = (cv::Mat_<float>(3, 1) << pos(0), pos(1), pos(2));
    cv::Mat EP = (cv::Mat_<float>(3, 1) << pos(3), pos(4), pos(5));
    cv::Mat MP = 0.5 * (SP + EP);

    cv::Mat CM = MP - pRefKF->GetCameraCenter();
    const float dist = cv::norm(CM);
    const int level = pRefKF->mvKeyLines[observations[pRefKF]].octave;
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal / n;
    }
}


float MapLine::GetMinDistanceInvariance() {
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f * mfMinDistance;
}


float MapLine::GetMaxDistanceInvariance() {
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f * mfMaxDistance;
}


int MapLine::PredictScale(const float &currentDist, const float &logScaleFactor) {
    float ratio;
    {
        unique_lock<mutex> lock3(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }
    return ceil(log(ratio) / logScaleFactor);
}

}
