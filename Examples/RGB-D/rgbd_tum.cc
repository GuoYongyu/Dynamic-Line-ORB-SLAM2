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

/**
 * List of Dataset Folders & Files:
 * - Dataset Root Directory
 *   + rgb
 *   + depth
 *   - rgb.txt
 *   - depth.txt
 *   - association.txt
 * 
 * Data Preprocessing Command: python associate.py PATH_TO_DATASET/rgb.txt PATH_TO_DATASET/depth.txt > association.txt
 * Launch Command: ./Examples/RGB-D/rgbd-tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml YoloConfig YoloConfig/yolov5_config.yaml PATH_TO_DATASET ASSOCIATION_FILE
 */


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>

#include <System.h>


using namespace std;


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);


int main(int argc, char **argv) {
    if (argc != 7) {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_yolo path_to_yolo_config path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[6]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = int(vstrImageFilenamesRGB.size());
    if (vstrImageFilenamesRGB.empty()) {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    } else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size()) {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], argv[3], argv[4], ORB_SLAM2::System::RGBD, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for (int ni = 0; ni < nImages; ni++) {
        cout << endl << "Pricessing image No." << ni + 1 << "/" << nImages << endl;

        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[5]) + "/" + vstrImageFilenamesRGB[ni], CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[5]) + "/" + vstrImageFilenamesD[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imRGB.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(argv[5]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB, imD, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = float(ttrack);
        cout << "(Main) Tracking time: " << ttrack * 1000 << "ms" << endl;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep(int((T - ttrack) * double(1e6)));
        
        // if (ni >= 100) {  // test test test
        //     break;
        // }
        SLAM.SaveMapFeatures2D("sparse-" + to_string(ni + 1) + ".txt");
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totalTime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        totalTime += vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totalTime / float(nImages) << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveMapFeatures("MapFeatures(Point-Line).txt");

    return 0;
}


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps) {
    cout << "Loading association file..." << endl;
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while (!fAssociation.eof()) {
        string s;
        getline(fAssociation,s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
    cout << "Finished load association file." << endl << endl;
}
