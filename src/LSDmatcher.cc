#include "LSDmatcher.h"

using namespace std;


namespace ORB_SLAM2 {

const int LSDmatcher::TH_HIGH = 100;
const int LSDmatcher::TH_LOW = 50;
const int LSDmatcher::HISTO_LENGTH = 30;


LSDmatcher::LSDmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri) {}


int LSDmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame) {
    // Return the number of matched line features 
    int nmatches = 0;
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    Mat ldesc1, ldesc2;
    vector<vector<DMatch>> lmatches;
    ldesc1 = LastFrame.mLdesc;
    ldesc2 = CurrentFrame.mLdesc;
    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

    double nn_dist_th, nn12_dist_th;
    CurrentFrame.LineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
    nn12_dist_th = nn12_dist_th*0.5;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    for (int i = 0; i < lmatches.size(); i++) {
        int qdx = lmatches[i][0].queryIdx;
        int tdx = lmatches[i][0].trainIdx;
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if (dist_12 > nn12_dist_th) {
            MapLine* mapLine = LastFrame.mvpMapLines[qdx];
            nmatches++;
        }
    }
    return nmatches;
}


int LSDmatcher::SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th) {
    int nmatches = 0;

    const bool bFactor = th != 1.0;

    for (size_t iML = 0; iML < vpMapLines.size(); iML++) {
        MapLine* pML = vpMapLines[iML];

        // Check whether the line need to be projected
        if (!pML->mbTrackInView)
            continue;

        if (pML->isBad())
            continue;

        // Predict level of pyramid, level is related current frame
        const int &nPredictLevel = pML->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        // When viewing angle and average viewing angle is too small, r will be close to 0
        float r = RadiusByViewingCos(pML->mTrackViewCos);

        // Increase range when coarser search is needed
        if (bFactor)
            r *= th;

        // Find line segments that can be matched
        vector<size_t> vIndices =
                F.GetLinesInArea(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2,
                                 r * F.mvScaleFactors[nPredictLevel], nPredictLevel - 1, nPredictLevel);

        if (vIndices.empty())
            continue;

        const cv::Mat MLdescriptor = pML->GetDescriptor();

        int bestDist = 256;
        int bestLevel = -1;
        int bestDist2 = 256;
        int bestLevel2 = -1;
        int bestIdx = -1 ;

        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
            const size_t idx = *vit;

            if (F.mvpMapLines[idx])
                if (F.mvpMapLines[idx]->Observations()>0)
                    continue;

            const cv::Mat &d = F.mLdesc.row(idx);

            const int dist = DescriptorDistance(MLdescriptor, d);

            // According descriptor, find the first and second closest feature
            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeyLinesUn[idx].octave;
                bestIdx = idx;
            } else if (dist < bestDist2) {
                bestLevel2 = F.mvKeyLinesUn[idx].octave;
                bestDist2 = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if (bestDist <= TH_HIGH) {
            if(bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                continue;

            F.mvpMapLines[bestIdx] = pML;
            nmatches++;
        }
    }
    return nmatches;
}


int LSDmatcher::SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, vector<pair<int, int>> &LineMatches) {
    LineMatches.clear();
    int nmatches = 0;
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    Mat ldesc1, ldesc2;
    vector<vector<DMatch>> lmatches;
    ldesc1 = InitialFrame.mLdesc;
    ldesc2 = CurrentFrame.mLdesc;
    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

    double nn_dist_th, nn12_dist_th;
    CurrentFrame.LineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
    nn12_dist_th = nn12_dist_th*0.5;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    for (int i = 0; i < lmatches.size(); i++) {
        int qdx = lmatches[i][0].queryIdx;
        int tdx = lmatches[i][0].trainIdx;
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if (dist_12 > nn12_dist_th) {
            LineMatches.push_back(make_pair(qdx, tdx));
            nmatches++;
        }
    }
    return nmatches;
}


int LSDmatcher::DescriptorDistance(const Mat &a, const Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


int LSDmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                       vector<pair<size_t, size_t>> &vMatchedPairs) {
    vMatchedPairs.clear();
    int nmatches = 0;
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    Mat ldesc1, ldesc2;
    vector<vector<DMatch>> lmatches;
    ldesc1 = pKF1->mLineDescriptors;
    ldesc2 = pKF2->mLineDescriptors;
    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

    double nn_dist_th, nn12_dist_th;
    pKF1->LineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
    nn12_dist_th = nn12_dist_th*0.1;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    for (int i = 0; i < lmatches.size(); i++) {
        int qdx = lmatches[i][0].queryIdx;
        int tdx = lmatches[i][0].trainIdx;
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if (dist_12 > nn12_dist_th) {
            vMatchedPairs.push_back(make_pair(qdx, tdx));
            nmatches++;
        }
    }
    return nmatches;
}


int LSDmatcher::Fuse(KeyFrame *pKF, const vector<MapLine*> &vpMapLines) {
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    Mat lineDesc = pKF->mLineDescriptors;
    const int nMLs = vpMapLines.size();

    for (int i = 0; i < nMLs; i++) {
        MapLine* pML = vpMapLines[i];

        if (!pML)
            continue;

        if (pML->isBad() || pML->IsInKeyFrame(pKF))
            continue;
#if 0
        Vector6d LineW = pML->GetWorldPos();
        cv::Mat LineSW = (Mat_<float>(3, 1) << LineW(0), LineW(1), LineW(2));
        cv::Mat LineSC = Rcw*LineSW + tcw;
        cv::Mat LineEW = (Mat_<float>(3, 1) << LineW(3), LineW(4), LineW(5));
        cv::Mat LineEC = Rcw*LineEW + tcw;

        // Depth must be positive
        if (LineSC.at<float>(2) < 0.0f || LineEC.at<float>(2) < 0.0f)
            continue;

        // 获取起始点在图像上的投影坐标
        const float invz1 = 1 / LineSC.at<float>(2);
        const float x1 = LineSC.at<float>(0) * invz1;
        const float y1 = LineSC.at<float>(1) * invz1;

        const float u1 = fx*x1 + cx;
        const float v1 = fy*y1 + cy;

        // 获取终止点在图像上的投影坐标
        const float invz2 = 1 / LineEC.at<float>(2);
        const float x2 = LineEC.at<float>(0) * invz2;
        const float y2 = LineEC.at<float>(1) * invz2;

        const float u2 = fx * x2 + cx;
        const float v2 = fy * y2 + cy;
#endif
        // The descriptro of MapLine[i]
        Mat CurrentLineDesc = pML->mLDescriptor;        

#if 0
        // 采用暴力匹配法, knnMatch
        BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
        vector<vector<DMatch>> lmatches;
        bfm->knnMatch(CurrentLineDesc, lineDesc, lmatches, 2);
        double nn_dist_th, nn12_dist_th;
        pKF->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th * 0.1;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for (int i = 0; i < lmatches.size(); i++) {
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if (dist_12 > nn12_dist_th) {  // Found ML in KF
                MapLine* pMLinKF = pKF->GetMapLine(tdx);
                if (pMLinKF) {
                    if (!pMLinKF->isBad()) {
                        if(pMLinKF->Observations() > pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                nFused++;
            }
        }
#elif 1
        // 使用暴力匹配法
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        vector<DMatch> lmatches;
        matcher->match(CurrentLineDesc, lineDesc, lmatches);

        double max_dist = 0;
        double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for (int i = 0; i < CurrentLineDesc.rows; i++) {
            double dist = lmatches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        // "good" matches (i.e. whose distance is less than 2*min_dist / 1.5*min_dist )
        std::vector<DMatch> good_matches;
        for (int i = 0; i < CurrentLineDesc.rows; i++) {
            if (lmatches[i].distance < 2 * min_dist) {
                int tdx = lmatches[i].trainIdx;
                good_matches.push_back(lmatches[i]);
                MapLine* pMLinKF = pKF->GetMapLine(tdx);
                if (pMLinKF) {
                    if (!pMLinKF->isBad()) {
                        if (pMLinKF->Observations() > pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                } else {
                    pML->AddObservation(pKF, tdx);
                    pKF->AddMapLine(pML, tdx);
                }
                nFused++;
            }
        }
        // cout << "(LSDmatcher) Find " << nFused << " matches" << endl;

#else
        cout << "CurrentLineDesc.empty() = " << CurrentLineDesc.empty() << endl;
        cout << "lineDesc.empty() = " << lineDesc.empty() << endl;
        cout << CurrentLineDesc << endl;
        if (CurrentLineDesc.empty() || lineDesc.empty())
            continue;

        // 采用Flann方法
        FlannBasedMatcher flm;
        vector<DMatch> lmatches;
        flm.match (CurrentLineDesc, lineDesc, lmatches);

        double max_dist = 0;
        double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        cout << "CurrentLineDesc.rows = " << CurrentLineDesc.rows << endl;
        for (int i = 0; i < CurrentLineDesc.rows; i++) { 
            double dist = lmatches[i].distance;
            if(dist < min_dist) min_dist = dist;
            if(dist > max_dist) max_dist = dist;
        }

        // "good" matches (i.e. whose distance is less than 2*min_dist )
        std::vector<DMatch> good_matches;
        for (int i = 0; i < CurrentLineDesc.rows; i++) {
            if (lmatches[i].distance < 2*min_dist) {
                int tdx = lmatches[i].trainIdx;
                good_matches.push_back(lmatches[i]);
                MapLine* pMLinKF = pKF->GetMapLine(tdx);
                if (pMLinKF) {
                    if (!pMLinKF->isBad()) {
                        if (pMLinKF->Observations() > pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                nFused++;
            }
        }
#endif
    }
    return nFused;
}


float LSDmatcher::RadiusByViewingCos(const float &viewCos) {
    if (viewCos > 0.998)
        return 5.0;
    else
        return 8.0;
}

}
