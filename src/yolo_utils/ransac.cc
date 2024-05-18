#include "include/yolo_utils/ransac.h"


template<typename T>
double getMean(const std::vector<T> &array, const std::vector<int> &indices) {
    int n = 0;
    double sum = 0;
    for (int i = 0; i < int(indices.size()); ++i) {
        if (indices[i] > 0) {
            sum += array[i];
            n++;
        }
    }
    return sum / n;
}


template<typename T>
double getVariance(const std::vector<T> &array, const std::vector<int> &indices, const double &mean) {
    int n = 0;
    double sum = 0;
    for (int i = 0; i < int(indices.size()); ++i) {
        if (indices[i] > 0) {
            sum += (array[i] - mean) * (array[i] - mean);
            n++;
        }
    }
    return sum / n;
}


std::vector<int> getRandomIndices(const int &n, int k) {
    std::vector<int> indices;
    for (int i = 0; i < n; ++i) {
        if (std::rand() % (n - i) < k) {
            indices.push_back(1);
            k--;
        } else {
            indices.push_back(0);
        }
    }
    return indices;
}


ushort getRansacDepth(const std::vector<ushort> &array,
                      double inliersRatio,
                      const int &minInliers,
                      const double &confidence,
                      const int &maxIters) {
    int iter = 0, kIters = 1;
    double bestError = 1.0e10;
    double depth = 1.0e10;
    inliersRatio = std::min(inliersRatio, double(minInliers) / double(array.size()) + 0.05);

    while (iter < std::min(kIters, maxIters)) {
        std::vector<int> maybeInliers = getRandomIndices(int(array.size()), minInliers);
        double mean = getMean(array, maybeInliers);
        double var = getVariance(array, maybeInliers, mean);

        for (int i = 0; i < int(array.size()); ++i) {
            if (maybeInliers[i] == 1) {
                continue;
            }
            maybeInliers[i] = 1;
            auto tmpMean = getMean(array, maybeInliers);
            auto tmpVar = getVariance(array, maybeInliers, mean);
            if (tmpVar < var && fabs(mean - tmpMean) < 0.01 * tmpMean) {
                mean = tmpMean;
                var = tmpVar;
            } else {
                maybeInliers[i] = 0;
            }
        }

        int nInliers = std::accumulate(maybeInliers.begin(), maybeInliers.end(), 0);
        if (nInliers >= int(inliersRatio * double(array.size()))) {
            // find suitable consensus inliers
            if (var < bestError && mean < depth) {
                depth = mean;
                bestError = var;
            }
        }

        double ratio = double(nInliers) / double(array.size());
        kIters = int(log(1 - confidence) / log(1 - pow(ratio, minInliers)));

        iter++;
    }

    return ushort(depth);
}
