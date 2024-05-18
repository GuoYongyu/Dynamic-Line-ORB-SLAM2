//
// Created by Star Rain on 2023/5/24.
//

#ifndef SEMANTICMASK_RANSAC_H
#define SEMANTICMASK_RANSAC_H


#include <iostream>
#include <algorithm>
#include <vector>
#include <iterator>
#include <numeric>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>


std::vector<int> getRandomIndices(const int& n, int k);

ushort getRansacDepth(const std::vector<ushort>& array,
                      double inliersRatio = 0.5,
                      const int& minInliers = 10,
                      const double& confidence = 0.95,
                      const int& maxIters = 100);


#endif //SEMANTICMASK_RANSAC_H
