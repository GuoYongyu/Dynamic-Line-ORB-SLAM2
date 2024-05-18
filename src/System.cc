/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>

namespace ORB_SLAM2 {

System::System(const string &strVocFile, const string &strSettingsFile,
               const string &strYoloConfigPath, const string &strYoloConfigFile,
               const eSensor sensor, const bool bUseViewer):
               mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), 
               mbReset(false), mbActivateLocalizationMode(false),
               mbDeactivateLocalizationMode(false) {
    // Output welcome message
    cout << endl << "Input sensor was set to: ";

    if(mSensor == MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor == STEREO)
        cout << "Stereo" << endl;
    else if(mSensor == RGBD)
        cout << "RGB-D" << endl;

    // Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    // Initialize YOLOv5 detector
    yolov5::Yolov5Config::setParameterFile(strYoloConfigFile);
    yolov5::Yolov5Config::setRootPath(strYoloConfigPath);
    mYoloType = yolov5::parseYoloModelType(yolov5::Yolov5Config::get<string>("yolov5_type"));
    // mpYoloDetector = new yolov5::Yolov5Detector(yoloType);

    // Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if (!bVocLoad) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    // Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    // Create the Map
    mpMap = new Map();

    // Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    // Initialize PointCloud mapping
    float resolution = fsSettings["PointCloudMap.Resolution"];
    mpPointCloudMap = make_shared<PointCloudMap>(resolution);

    // Initialize the Tracking thread
    // (it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mYoloType, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpPointCloudMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    // Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

    // Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

    // Initialize the Viewer thread and launch
    if (bUseViewer) {
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
        // mptViewer = new thread(&Viewer::Run, mpViewer);  // Points only
        mptViewer = new thread(&Viewer::RunWithLine, mpViewer);  // Points & Lines
        mpTracker->SetViewer(mpViewer);
    }

    // Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp) {
    if (mSensor != STEREO) {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped()) {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset) {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp) {
    if (mSensor != RGBD) {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while (!mpLocalMapper->isStopped()) {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset) {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    mTrackedMapLines = mpTracker->mCurrentFrame.mvpMapLines;
    mTrackedKeyLines = mpTracker->mCurrentFrame.mvKeyLinesUn;
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp) {
    if (mSensor != MONOCULAR) {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while (!mpLocalMapper->isStopped()) {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset) {
            mpTracker->Reset();
            mbReset = false;
        }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

void System::ActivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged() {
    static int n = 0;
    int curn = mpMap->GetLastBigChangeIdx();
    if (n < curn) {
        n = curn;
        return true;
    }
    else
        return false;
}

void System::Reset() {
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown() {
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    mpPointCloudMap->shutdown();  // stop point cloud mapping

    if (mpViewer) {
        mpViewer->RequestFinish();
        while (!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA()) {
        usleep(5000);
    }

    if (mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename) {
    cout << endl << "(System) Saving camera trajectory to Output/" << filename << " ..." << endl;
    if (mSensor == MONOCULAR) {
        cerr << "(System) ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(("./Output/" + filename).c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
        lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++) {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while (pKF->isBad()) {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        cv::Mat Tcw = (*lit) * Trw;
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) 
          << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) 
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "(System) Trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename) {
    cout << endl << "(System) Saving keyframe trajectory to Output/" << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(("./Output/" + filename).c_str());
    f << fixed;

    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if (pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) 
          << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "(System) Keyframe's trajectory saved!" << endl;
}


void System::SaveMapFeatures(const string &filename) {
    cout << endl << "(System) Saving point/line features to " << filename << " ..." << endl;

    ofstream f;
    f.open(("./Output/" + filename).c_str());
    f << fixed;
    f << "#point x_w y_w z_w" << endl;
    f << "#line sx_w sy_w sz_w ex_w ey_w ez_w r g b label" << endl;

    vector<MapPoint*> vpPoints = mpMap->GetAllMapPoints();
    for (size_t i = 0; i < vpPoints.size(); ++i) {
        MapPoint* pPoint = vpPoints[i];
        if (pPoint->isBad()) {
            continue;
        }

        cv::Mat pos = pPoint->GetWorldPos();
        f << "point " << setprecision(7) << " " 
          << pos.at<float>(0) << " " << pos.at<float>(1) << " " << pos.at<float>(2) << endl;
    }

    vector<MapLine*> vpLines = mpMap->GetAllMapLines();
    for (size_t i = 0; i < vpLines.size(); i++) {
        MapLine* pLine = vpLines[i];
        if (pLine->isBad()) {
            continue;
        }

        Vector6d pos = pLine->GetWorldPos();
        string label = pLine->msSemanticLabel;
        f << "line " << setprecision(7) << " "
          << pos(0) << " " << pos(1) << " " << pos(2) << " "
          << pos(3) << " " << pos(4) << " " << pos(5) << " "
          << yolov5::Yolov5Detector::planarColorMap[label][0] << " "
          << yolov5::Yolov5Detector::planarColorMap[label][1] << " "
          << yolov5::Yolov5Detector::planarColorMap[label][2] << " "
          << label << endl;
    }

    cout << endl << "(System) Map features (Points & Lines) saved!" << endl;
}


void System::SaveMapFeatures2D(const string &filename) {
    cout << endl << "(System) Saving point/line features to " << filename << " ..." << endl;

    ofstream f;
    f.open(("./Output/" + filename).c_str());
    f << fixed;
    f << "#point u_c v_c" << endl;
    f << "#line s_u s_v e_u e_v r g b label" << endl;

    vector<cv::KeyPoint> vPoints = mpTracker->mCurrentFrame.mvKeysUn;
    for (cv::KeyPoint& point: vPoints) {
        f << "point " << setprecision(7) << " "
          << point.pt.x << " " << point.pt.y << endl;
    }

    vector<cv::line_descriptor::KeyLine> vLines = mpTracker->mCurrentFrame.mvKeyLinesUn;
    for (cv::line_descriptor::KeyLine& line: vLines) {
        string label = getLineLabelInBox(
            mpTracker->mCurrentFrame.mWidth, mpTracker->mCurrentFrame.mHeight,
            mpTracker->mCurrentFrame.mvPlanarBoxes, line,
            mpTracker->mCurrentFrame.mPlanarBoxExpansionRate,
            mpTracker->mCurrentFrame.mLineIntersectionRate
        );
        f << "line " << setprecision(7) << " "
          << line.startPointX << " " << line.startPointY << " "
          << line.endPointX << " " << line.endPointY << " "
          << yolov5::Yolov5Detector::planarColorMap[label][0] << " "
          << yolov5::Yolov5Detector::planarColorMap[label][1] << " "
          << yolov5::Yolov5Detector::planarColorMap[label][2] << " "
          << label << endl;
    }

    cout << endl << "(System) 2D features (Points & Lines) saved!" << endl;
}


void System::SaveTrajectoryKITTI(const string &filename) {
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if (mSensor == MONOCULAR) {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(), 
        lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++) {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

        while (pKF->isBad()) {
          //  cout << "bad parent" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        cv::Mat Tcw = (*lit) * Trw;
        cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0, 3).col(3);

        f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1)  << " " << Rwc.at<float>(0, 2) << " "  << twc.at<float>(0) << " "
          << Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1)  << " " << Rwc.at<float>(1, 2) << " "  << twc.at<float>(1) << " "
          << Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1)  << " " << Rwc.at<float>(2, 2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
