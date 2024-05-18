﻿/**
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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

namespace ORB_SLAM2
{


void Optimizer::GlobalBundleAdjustemnt(
        Map* pMap, int nIterations, const bool bWithLineFeature, 
        bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust) {
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    if (bWithLineFeature) {
        vector<MapLine*> vpML = pMap->GetAllMapLines();
        cout << "(Global BA) BA with points and lines." << endl;
        BundleAdjustment(vpKFs, vpMP, vpML, nIterations, pbStopFlag, nLoopKF, bRobust);
    } else {
        cout << "(Global BA) BA with only points." << endl;
        BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
    }
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust) {
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId == 0);
        optimizer.addVertex(vSE3);
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); i++) {
        MapPoint* pMP = vpMP[i];
        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const map<KeyFrame*, size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
            KeyFrame* pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if (pKF->mvuRight[mit->second] < 0) {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            } else {
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if (nEdges == 0) {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        } else {
            vbNotIncludedMP[i] = false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == 0) {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        } else {
            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for (size_t i = 0; i < vpMP.size(); i++) {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));

        if (nLoopKF == 0) {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        } else {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

/**
 * @brief 包含线特征的BA
 * @param vpKFs 关键帧
 * @param vpMP MapPoints
 * @param vpML MapLines
 * @param nIterations 迭代次数，20次
 * @param pbStopFlag 是否强制暂停
 * @param nLoopKF 关键帧的个数
 * @param bRobust 是否使用核函数
 *
 * 3D-2D 最小化重投影误差 e = (u, v) - project(Tcw * Pw)
 * 修改为：e = points的重投影误差 + lines的重投影误差
 *          = (u, v)-project(Tcw * Pw) + line * project(Tcw * LineStartPointW) + line * project(Tcw * LineEndPointW)
 * 1.Vertex:  g2o::VertexSE3Expmap(), 即当前帧的Tcw
 *            g2o::VertexSBAPointXYZ(), MapPoint的mWorldPos
 *            g2o::VertexLinePointXYZ(), MapLine的端点世界坐标
 * 2.Edge:
 *      - g2o::EdgeSE3ProjectXYZ(): public BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>
 *          + Vertex：待优化当前帧的Tcw
 *          + measurement: MapPoint在当前帧中的二维位置(u, v)
 *          + InfoMatrix：invSigma2（与特征点所在的尺度有关）
 *      - EdgeLineProjectXYZ: public BaseBinaryEdge<3, Vector3d, g2o::VertexSE3Expmap, g2o::VertexLinePointXYZ>
 *          + Vertex1: 待优化当前帧的Tcw
 *          + Vertex2: 待优化的MapLine的一个端点
 *          + measuremtn: MapLine的一个端点在当前帧中的二维位置(u, v)
 */
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, const vector<MapLine *> &vpML,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust) {
    double invSigma = 0.01;
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    // 1. 构造求解器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    // 2. 添加关键帧顶点
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId == 0);
        optimizer.addVertex(vSE3);
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    vector<int> MapPointID;

    // 3.设置MapPoint的顶点
    for (size_t i = 0; i < vpMP.size(); i++) {
        MapPoint* pMP = vpMP[i];
        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        MapPointID.push_back(id);

        const map<KeyFrame*, size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        // 4.设置MapPoint和位姿之间的误差边
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
            KeyFrame* pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if (pKF->mvuRight[mit->second] < 0) {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            } else {  //双目
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust) {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if (nEdges == 0) {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        } else {
            vbNotIncludedMP[i] = false;
        }
    }

    sort(MapPointID.begin(), MapPointID.end());
    int maxMapPointID = MapPointID[MapPointID.size() - 1];

    // 5. 设置MapLine的顶点，线段有两个端点，因此顶点数量和对应的边的数量为MapLine数量的2倍
    // 5.1 先设置线段的起始点
    for (size_t i = 0; i < vpML.size(); i++) {
        MapLine* pML = vpML[i];
        if (pML->isBad())
            continue;
        // VertexLinePointXYZ* vStartP = new VertexLinePointXYZ();   // 用自定义的顶点
        g2o::VertexSBAPointXYZ* vStartP = new g2o::VertexSBAPointXYZ();
        vStartP->setEstimate(pML->GetWorldPos().head(3));
        const int ids = pML->mnId + maxKFid + maxMapPointID + 1;
        vStartP->setId(ids);
        vStartP->setMarginalized(true);
        optimizer.addVertex(vStartP);

        const map<KeyFrame*,size_t> observations = pML->mObservations;

        int nLineEdges = 0;

        // 设置线段起始点和相机位姿之间的边
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
            KeyFrame* pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nLineEdges++;

            Eigen::Vector3d line_obs;
            line_obs = pKF->mvKeyLineFunctions[mit->second];

            EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ids)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(line_obs);
            e->setInformation(Eigen::Matrix3d::Identity() * invSigma);

            if (bRobust) {
                g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk_line_s);
                rk_line_s->setDelta(thHuber2D);
            }

            e->fx = pKF->fx;
            e->fy = pKF->fy;
            e->cx = pKF->cx;
            e->cy = pKF->cy;

            e->Xw = pML->mWorldPos.head(3);

            optimizer.addEdge(e);
        }
    }

    // 5.2 设置线段终止点顶点
    for (size_t i = 0; i < vpML.size(); i++) {
        MapLine* pML = vpML[i];
        if (pML->isBad())
            continue;
        // VertexLinePointXYZ* vEndP = new VertexLinePointXYZ();     // 自定义的顶点
        g2o::VertexSBAPointXYZ* vEndP = new VertexSBAPointXYZ();
        vEndP->setEstimate(pML->GetWorldPos().tail(3));
        const int ide = pML->mnId + maxKFid + maxMapPointID + vpML.size() + 1;
        vEndP->setId(ide);
        vEndP->setMarginalized(true);
        optimizer.addVertex(vEndP);

        const map<KeyFrame*,size_t> observations = pML->mObservations;

        int nLineEdges = 0;

        // 设置线段终止点和相机位姿之间的边
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
            KeyFrame* pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nLineEdges++;

            Eigen::Vector3d line_obs;
            line_obs = pKF->mvKeyLineFunctions[mit->second];

            EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ide)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(line_obs);
            e->setInformation(Eigen::Matrix3d::Identity() * invSigma);

            if (bRobust) {
                g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk_line_s);
                rk_line_s->setDelta(thHuber2D);
            }

            e->fx = pKF->fx;
            e->fy = pKF->fy;
            e->cx = pKF->cx;
            e->cy = pKF->cy;

            e->Xw = pML->mWorldPos.head(3);

            optimizer.addEdge(e);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    // 6.得到优化的结果

    // Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == 0) {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        } else {
            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    // Points
    for (size_t i = 0; i < vpMP.size(); i++) {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));

        if (nLoopKF == 0) {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        } else {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

    // LineSegment Points
    for (size_t i = 0; i < vpML.size(); i++) {
        if (vbNotIncludedMP[i])
            continue;

        MapLine* pML = vpML[i];

        if (pML->isBad())
            continue;

        // VertexLinePointXYZ* vStartP = static_cast<VertexLinePointXYZ*>(optimizer.vertex(pML->mnId + maxKFid + vpMP.size() + 1));  //自定义的顶点
        // VertexLinePointXYZ* vEndP = static_cast<VertexLinePointXYZ*>(optimizer.vertex(2*pML->mnId + maxKFid + vpMP.size() + 1));  //自定义的顶点

        g2o::VertexSBAPointXYZ* vStartP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + 1));
        g2o::VertexSBAPointXYZ* vEndP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + vpML.size() + 1));

        if (nLoopKF == 0) {
            Vector6d LinePos;
            LinePos << Converter::toVector3d(Converter::toCvMat(vStartP->estimate())), Converter::toVector3d(Converter::toCvMat(vEndP->estimate()));
            pML->SetWorldPos(LinePos);
            pML->UpdateAverageDir();
        } else {
            pML->mPosGBA.create(6, 1, CV_32F);
            Converter::toCvMat(vStartP->estimate()).copyTo(pML->mPosGBA.rowRange(0, 3));
            Converter::toCvMat(vEndP->estimate()).copyTo(pML->mPosGBA.rowRange(3, 6));
        }
    }
}

/**
 * 该优化函数主要用于Tracking线程中
 *
 * 3D-2D 最小化重投影误差 e = (u, v) - project(Tcw * Pw)
 * 修改为： e = points的重投影误差 + lines的重投影误差
 *          = (u, v) - project(Tcw*Pw) + line * project(Tcw * LineStartPointw) + line * project(Tcw * LineEndPointW)
 * 只优化pFrame的Tcw，不优化MapPoints和MapLines的坐标
 *
 * 1.Vertex: g2o::VertexSE3Expmap，即当前帧的Tcw
 * 2.Edge:
 *      - g2o::EdgeSE3ProjectXYZOnlyPose: public BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>
 *          + Vertex：待优化当前帧的Tcw
 *          + measurement：MapPoint在当前帧中的二维位置(u, v)
 *          + InfoMatrix：invSigma2（与特征点所在的尺度有关）
 *
 * @param pFrame 图像帧
 * @return
 */
int Optimizer::PoseOptimization(Frame *pFrame) {
    double invSigma = 10;
    // 1.构造求解器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();  //这里用的是Dense，可能会影响速度

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++) {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (pMP) { // Monocular observation
                if (pFrame->mvuRight[i] < 0) {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                } else { // Stereo observation
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    //SET EDGE
                    Eigen::Matrix<double, 3, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }

        }
    }

    // 添加相机位姿和特征线段之间的误差边
    // Set MapLine vertices
    const int NL = pFrame->NL;
    int nLineInitalCorrespondences=0;

    // 起始点
    vector<EdgeLineProjectXYZOnlyPose*> vpEdgesLineSp;
    vector<size_t> vnIndexLineEdgeSp;
    vpEdgesLineSp.reserve(NL);

    // 终止点
    vector<EdgeLineProjectXYZOnlyPose*> vpEdgesLineEp;
    vector<size_t> vnIndexLineEdgeEp;
    vpEdgesLineEp.reserve(NL);
    vnIndexLineEdgeEp.reserve(NL);

    {
        unique_lock<mutex> lock(MapLine::mGlobalMutex);

        for (int i = 0; i < NL; i++) {
            MapLine* pML = pFrame->mvpMapLines[i];
            if (pML) {
                nLineInitalCorrespondences++;
                pFrame->mvbLineOutlier[i] = false;

                Eigen::Vector3d line_obs;
                line_obs = pFrame->mvKeyLineFunctions[i];

                // 特征线段的起始点
                EdgeLineProjectXYZOnlyPose* els = new EdgeLineProjectXYZOnlyPose();

                els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                els->setMeasurement(line_obs);
                els->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                els->setRobustKernel(rk_line_s);
                rk_line_s->setDelta(deltaMono);

                els->fx = pFrame->fx;
                els->fy = pFrame->fy;
                els->cx = pFrame->cx;
                els->cy = pFrame->cy;

                els->Xw = pML->mWorldPos.head(3);

                optimizer.addEdge(els);

                vpEdgesLineSp.push_back(els);
                vnIndexLineEdgeSp.push_back(i);

                // 特征点的终止点
                EdgeLineProjectXYZOnlyPose* ele = new EdgeLineProjectXYZOnlyPose();

                ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                ele->setMeasurement(line_obs);
                ele->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                g2o::RobustKernelHuber* rk_line_e = new g2o::RobustKernelHuber;
                ele->setRobustKernel(rk_line_e);
                rk_line_e->setDelta(deltaMono);

                ele->fx = pFrame->fx;
                ele->fy = pFrame->fy;
                ele->cx = pFrame->cx;
                ele->cy = pFrame->cy;

                ele->Xw = pML->mWorldPos.tail(3);

                optimizer.addEdge(ele);

                vpEdgesLineEp.push_back(ele);
                vnIndexLineEdgeEp.push_back(i);
            }
        }
    }

    if (nInitialCorrespondences < 3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
    const int its[4] = {10, 10, 10, 10};     

    int nBad = 0;     //点特征
    int nLineBad = 0; //线特征
    for(size_t it = 0; it < 4; it++) {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->mvbOutlier[idx]) {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2>chi2Mono[it]) {                
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            } else {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx]) {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it]) {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            } else {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx] = false;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        nLineBad = 0;
        for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
            EdgeLineProjectXYZOnlyPose* e1 = vpEdgesLineSp[i];  //线段起始点
            EdgeLineProjectXYZOnlyPose* e2 = vpEdgesLineEp[i];  //线段终止点

            const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

            if (pFrame->mvbLineOutlier[idx]) {
                e1->computeError();
                e2->computeError();
            }

            const float chi2_s = e1->chi2();
            const float chi2_e = e2->chi2();

            if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                pFrame->mvbLineOutlier[idx] = true;
                e1->setLevel(1);
                e2->setLevel(1);
                nLineBad++;
            } else {
                pFrame->mvbLineOutlier[idx] = false;
                e1->setLevel(0);
                e2->setLevel(0);
            }

            if (it == 2) {
                e1->setRobustKernel(0);
                e2->setRobustKernel(0);
            }
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences - nBad;
}

/**
 * @brief Local BA
 *
 * 1.Vertex:
 *      - g2o::VertexSE3Expmap，LocalKeyFrames，即当前关键帧的位姿，与当前关键帧相连的关键帧的位姿
 *      - g2o::VertexSE3Expmap，FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocalMappingKeyFrames）的位姿，在优化中这些关键帧的位姿不变
 *      - g2o：：VertexSBAPointXYZ()，LocalMapPoints, 即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2.Edge:
 *      -g2o::EdgeSE3ProjectXYZ(), BaseBinaryEdge
 *          + Vertex：关键帧的Tcw，MapPoint的Pw
 *          + measurement:MapPoint在关键帧中的二维位置(u, v)
 *          + InforMatrix：invSigma2,与特征点所在的尺度有关
 *      -g2o::EdgeStereoSE3ProjectXYZ(), BaseBinaryEdge
 *          + Vertex：关键帧的Tcw, MapPoint的Pw
 *          + measurement: MapPoint在关键帧中的二维位置（ul, v, ur）
 *          + InfoMatrix： invSigma2，与特征点所在的尺度有关
 *
 * @param pKF   KeyFrame
 * @param pbStopFlag    是否停止优化的标识
 * @param pMap          在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap) {    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    // step1：将当前关键帧加入到lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    // step2：找到关键帧连接的关键帧（一级相连），加入lLocalKeyFrames中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++) {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    // step3：遍历lLocalKeyFrames中关键帧，将它们观测的MapPoints加入到lLocalMapPoints
    list<MapPoint*> lLocalMapPoints;
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for (vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
            MapPoint* pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId) {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // step4：得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时不优化
    list<KeyFrame*> lFixedCameras;
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
        for (map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
            KeyFrame* pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {                
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    // step5：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    // step6：添加顶点，Pose of Local KeyFrame
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == 0);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    // step7：添加顶点，Pose of Fixed KeyFrame，注意这里调用了vSE3->setFixed(true)
    for (list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set MapPoint vertices
    // step8：添加3D顶点
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*, size_t> observations = pMP->GetObservations();

        //Set edges
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
            KeyFrame* pKFi = mit->first;

            if (!pKFi->isBad()) {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if (pKFi->mvuRight[mit->second] < 0) {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                } else { // Stereo observation
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore) {
        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend;i++) {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive()) {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive()) {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame*,MapPoint*>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations       
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty()) {
        for (size_t i = 0; i < vToErase.size(); i++) {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    // Keyframes
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    // Points
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}

///包含有线特征的局部BA
void Optimizer::LocalBundleAdjustmentWithLine(KeyFrame *pKF, bool *pbStopFlag, Map *pMap) {
    double invSigma = 0.01;
    // Local KeyFrames: First Breath Search from Current KeyFrame
    list<KeyFrame*> lLocalKeyFrames;

    // step1: 将当前关键帧加入到lLocalKeyFrames
    // cout << "(Optimizer) LocalBAwithLines - step1" << endl;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    // step2:找到关键帧连接的关键帧（一级相连），加入到lLocalKeyFrames中
    // cout << "(Optimizer) LocalBAwithLines - step2" << endl;
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++) {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // step3：将lLocalKeyFrames的MapPoints加入到lLocalMapPoints
    // cout << "(Optimizer) LocalBAwithLines - step3" << endl;
    list<MapPoint*> lLocalMapPoints;
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        // cout << "(Optimizer) Frame id:" << (*lit)->mnFrameId << ", KeyFrame id: " << (*lit)->mnId << ", Pointss num: " << vpMPs.size() << endl;
        for (vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
            MapPoint* pMP = *vit;
            if (pMP) {
                if (!pMP->isBad()) {
                    if (pMP->mnBALocalForKF != pKF->mnId) {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                }
            }
        }
    }

    // step4: 遍历lLocalKeyFrames，将每个关键帧所能观测到的MapLine提取出来，放到lLocalMapLines
    // cout << "(Optimizer) LocalBAwithLines - step4" << endl;
    list<MapLine*> lLocalMapLines;
    vector<long unsigned int> mlID;
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        vector<MapLine*> vpMLs = (*lit)->GetMapLineMatches();
        // cout << "(Optimizer) Frame id:" << (*lit)->mnFrameId << ", KeyFrame id: " << (*lit)->mnId << ", Lines num: " << vpMLs.size() << endl;
        for (vector<MapLine*>::iterator vit = vpMLs.begin(), vend = vpMLs.end(); vit != vend; vit++) {
            MapLine* pML = *vit;
            if (pML) {
                if (!pML->isBad()) {
                    // cout << "(Optimizer) Not bad" << endl;
                    if (pML->mnBALocalForKF != pKF->mnId) {
                        lLocalMapLines.push_back(pML);
                        pML->mnBALocalForKF = pKF->mnId;
                        mlID.push_back(pML->mnId);
                    }
                }
            }
        }
    }

#if 0
    for (list<MapLine*>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
        int i = 0;
        if (count(mlID.begin(), mlID.end(), mlID[i]) > 1) {
            lit = lLocalMapLines.erase(lit);
        }
        i++;
    }
#endif

    // step5: 得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时固定
    // cout << "(Optimizer) LocalBAwithLines - step5" << endl;
    list<KeyFrame*> lFixedCameras;
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        map<KeyFrame*, size_t > observations = (*lit)->GetObservations();
        for (map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
            KeyFrame* pKFi = mit->first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // step6：构造g2o优化器
    // cout << "(Optimizer) LocalBAwithLines - step6" << endl;
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // step7：添加顶点，Pose of Local KeyFrame
    // cout << "(Optimizer) LocalBAwithLines - step7" << endl;
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == 0);  // 第一个关键帧需要Fix
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // step8:添加固定帧的顶点，Pose of Fixed KeyFrame
    // cout << "(Optimizer) LocalBAwithLines - step8" << endl;
    for (list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    vector<int> MapPointID;

    // step9：添加MapPoint的3D顶点
    // cout << "(Optimizer) LocalBAwithLines - step9" << endl;
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);

    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        MapPointID.push_back(id);

        const map<KeyFrame*, size_t > observations = pMP->GetObservations();

        // Set Edges
        for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
            KeyFrame* pKFi = mit->first;
            if (!pKFi->isBad()) {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                e->fx = pKFi->fx;
                e->fy = pKFi->fy;
                e->cx = pKFi->cx;
                e->cy = pKFi->cy;

                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFi);
                vpMapPointEdgeMono.push_back(pMP);
            }
        }
    }

    sort(MapPointID.begin(), MapPointID.end());
    int maxMapPointID = MapPointID[MapPointID.size() - 1];

    // step10：添加MapLine的顶点
    // cout << "(Optimizer) LocalBAwithLines - step10" << endl;
    const int nLineExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapLines.size();

    vector<EdgeLineProjectXYZ*> vpLineEdgesSP;
    vpLineEdgesSP.reserve(nLineExpectedSize);

    vector<EdgeLineProjectXYZ*> vpLineEdgesEP;
    vpLineEdgesEP.reserve(nLineExpectedSize);

    vector<KeyFrame*> vpLineEdgeKF;
    vpLineEdgeKF.reserve(nLineExpectedSize);

    vector<MapLine*> vpMapLineEdge;
    vpMapLineEdge.reserve(nLineExpectedSize);
    
    //cout << "(Optimizer) LocalMapLines size: " << lLocalMapLines.size() << endl;
    bool bUseLines = false;
    vector<int> MapLineSPID;
    if (0 != lLocalMapLines.size()) {
        bUseLines = true;
        
        // 起始点的顶点
        for (list<MapLine*>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
            MapLine* pML = *lit;
            g2o::VertexSBAPointXYZ* vStartP = new g2o::VertexSBAPointXYZ();
            vStartP->setEstimate(pML->GetWorldPos().head(3));
            int ids = pML->mnId + maxKFid + maxMapPointID + 1;
            vStartP->setId(ids);
            vStartP->setMarginalized(true);
            optimizer.addVertex(vStartP);
            MapLineSPID.push_back(ids);

            const map<KeyFrame*, size_t > observations = pML->GetObservations();

            for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
                KeyFrame* pKFi = mit->first;
                if (!pKFi->isBad()) {
                    Eigen::Vector3d line_obs;
                    line_obs = pKFi->mvKeyLineFunctions[mit->second];

                    EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ids)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(line_obs);
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    e->Xw = pML->mWorldPos.head(3);

                    optimizer.addEdge(e);
                    vpLineEdgesSP.push_back(e);
                    vpLineEdgeKF.push_back(pKFi);
                    vpMapLineEdge.push_back(pML);
                }
            }
        }

        sort(MapLineSPID.begin(), MapLineSPID.end());
        // cout << "(Optimizer) MapLineSPID size: " << MapLineSPID.size() << endl;
        int maxMapLineSPID = MapLineSPID[MapLineSPID.size() - 1];
        // 终止点的顶点
        for (list<MapLine*>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
            MapLine* pML = *lit;
            g2o::VertexSBAPointXYZ* vEndP = new VertexSBAPointXYZ();
            vEndP->setEstimate(pML->GetWorldPos().tail(3));
            int ide = pML->mnId + maxKFid + maxMapPointID + maxMapLineSPID + 1;
            vEndP->setId(ide);
            vEndP->setMarginalized(true);
            optimizer.addVertex(vEndP);

            const map<KeyFrame*, size_t> observations = pML->GetObservations();

            for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
                KeyFrame* pKFi = mit->first;
                if (!pKFi->isBad()) {
                    Eigen::Vector3d line_obs;
                    line_obs = pKFi->mvKeyLineFunctions[mit->second];

                    EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ide)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(line_obs);
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber* rk_line_e = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberMono);

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    e->Xw = pML->mWorldPos.head(3);

                    optimizer.addEdge(e);
                    vpLineEdgesEP.push_back(e);
                    vpLineEdgeKF.push_back(pKFi);
                }
            }
        }
    }
    
    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    // cout << "(Optimizer) LocalBAwithLines - optimize" << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if( bDoMore) {
        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive()) {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }

        if (bUseLines) {
            for (size_t i = 0, iend = vpLineEdgesSP.size(); i < iend; i++) {
                EdgeLineProjectXYZ* e = vpLineEdgesSP[i];
                MapLine* pML = vpMapLineEdge[i];

                if (pML->isBad())
                    continue;

                if (e->chi2() > 7.815) {
                    e->setLevel(1);
                }
                e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpLineEdgesEP.size(); i < iend; i++) {
                EdgeLineProjectXYZ* e = vpLineEdgesEP[i];
                MapLine* pML = vpMapLineEdge[i];

                if (pML->isBad())
                    continue;

                if (e->chi2() > 7.815) {
                    e->setLevel(1);
                }
                e->setRobustKernel(0);
            }
        }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame*, MapPoint*>> vToErase;
    vToErase.reserve(vpEdgesMono.size());

    // check inlier observations
    // cout << "(Optimizer) LocalBAwithLines - check inlier observations" << endl;
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive()) {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }
    
    vector<pair<KeyFrame*, MapLine*>> vLineToErase;
    if (bUseLines) {
        vLineToErase.reserve(vpLineEdgesSP.size());
        for (size_t i = 0, iend = vpLineEdgesSP.size(); i < iend; i++) {
            EdgeLineProjectXYZ* e = vpLineEdgesSP[i];
            MapLine* pML = vpMapLineEdge[i];

            if (pML->isBad())
                continue;

            if (e->chi2() > 7.815) {
                KeyFrame* pKFi = vpLineEdgeKF[i];
                vLineToErase.push_back(make_pair(pKFi, pML));
            }
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty()) {
        for (size_t i = 0; i < vToErase.size(); i++) {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    if (bUseLines) {
        if (!vLineToErase.empty()) {
            for (size_t i = 0; i < vLineToErase.size(); i++) {
                KeyFrame* pKFi = vLineToErase[i].first;
                MapLine* pMLi = vLineToErase[i].second;
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseObservation(pKFi);
            }
        }
    }

    // Recover optimized data
    // cout << "(Optimizer) LocalBAwithLines - recover optimized data" << endl;
    // Keyframes
    for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    // Points
    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    // Lines
    if (bUseLines) {
        int maxMapLineSPID = MapLineSPID[MapLineSPID.size() - 1];
        for (list<MapLine*>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
            MapLine* pML = *lit;
            g2o::VertexSBAPointXYZ* vStartP = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + 1));
            g2o::VertexSBAPointXYZ* vEndP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + maxMapLineSPID + 1));

            Vector6d LinePos;
            LinePos << Converter::toVector3d(Converter::toCvMat(vStartP->estimate())), Converter::toVector3d(Converter::toCvMat(vEndP->estimate()));
            pML->SetWorldPos(LinePos);
            pML->UpdateAverageDir();
        }
    }
}

///包含有线特征的局部BA
// void Optimizer::LocalBundleAdjustmentWithLine(KeyFrame *pKF, bool *pbStopFlag, Map *pMap) {
//     double invSigma = 0.01;
//     // Local KeyFrames: First Breath Search from Current KeyFrame
//     list<KeyFrame*> lLocalKeyFrames;

//     // step1: 将当前关键帧加入到lLocalKeyFrames
//     // cout << "(Optimizer) LocalBAwithLines - step1" << endl;
//     lLocalKeyFrames.push_back(pKF);
//     pKF->mnBALocalForKF = pKF->mnId;

//     // step2:找到关键帧连接的关键帧（一级相连），加入到lLocalKeyFrames中
//     // cout << "(Optimizer) LocalBAwithLines - step2" << endl;
//     const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
//     for (int i = 0, iend = vNeighKFs.size(); i < iend; i++) {
//         KeyFrame* pKFi = vNeighKFs[i];
//         pKFi->mnBALocalForKF = pKF->mnId;
//         if (!pKFi->isBad())
//             lLocalKeyFrames.push_back(pKFi);
//     }

//     // step3：将lLocalKeyFrames的MapPoints加入到lLocalMapPoints
//     // cout << "(Optimizer) LocalBAwithLines - step3" << endl;
//     list<MapPoint*> lLocalMapPoints;
//     for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
//         vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
//         // cout << "(Optimizer) Frame id:" << (*lit)->mnFrameId << ", KeyFrame id: " << (*lit)->mnId << ", Pointss num: " << vpMPs.size() << endl;
//         for (vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
//             MapPoint* pMP = *vit;
//             if (pMP) {
//                 if (!pMP->isBad()) {
//                     if (pMP->mnBALocalForKF != pKF->mnId) {
//                         lLocalMapPoints.push_back(pMP);
//                         pMP->mnBALocalForKF = pKF->mnId;
//                     }
//                 }
//             }
//         }
//     }

//     // step4: 遍历lLocalKeyFrames，将每个关键帧所能观测到的MapLine提取出来，放到lLocalMapLines
//     // cout << "(Optimizer) LocalBAwithLines - step4" << endl;
//     list<MapLine*> lLocalMapLines;
//     vector<long unsigned int> mlID;
//     for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
//         vector<MapLine*> vpMLs = (*lit)->GetMapLineMatches();
//         cout << "(Optimizer) Frame id:" << (*lit)->mnFrameId << ", KeyFrame id: " << (*lit)->mnId << ", Lines num: " << vpMLs.size() << endl;
//         for (vector<MapLine*>::iterator vit = vpMLs.begin(), vend = vpMLs.end(); vit != vend; vit++) {
//             MapLine* pML = *vit;
//             if (pML) {
//                 if (!pML->isBad()) {
//                     // cout << "(Optimizer) Not bad" << endl;
//                     if (pML->mnBALocalForKF != pKF->mnId) {
//                         lLocalMapLines.push_back(pML);
//                         pML->mnBALocalForKF = pKF->mnId;
//                         mlID.push_back(pML->mnId);
//                     }
//                 }
//             }
//         }
//     }

// #if 0
//     for (list<MapLine*>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
//         int i = 0;
//         if (count(mlID.begin(), mlID.end(), mlID[i]) > 1) {
//             lit = lLocalMapLines.erase(lit);
//         }
//         i++;
//     }
// #endif

//     // step5: 得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时固定
//     // cout << "(Optimizer) LocalBAwithLines - step5" << endl;
//     list<KeyFrame*> lFixedCameras;
//     for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
//         map<KeyFrame*, size_t > observations = (*lit)->GetObservations();
//         for (map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
//             KeyFrame* pKFi = mit->first;
//             if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) {
//                 pKFi->mnBAFixedForKF = pKF->mnId;
//                 if (!pKFi->isBad())
//                     lFixedCameras.push_back(pKFi);
//             }
//         }
//     }

//     // step6：构造g2o优化器
//     // cout << "(Optimizer) LocalBAwithLines - step6" << endl;
//     g2o::SparseOptimizer optimizer;
//     g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

//     linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

//     g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

//     g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//     optimizer.setAlgorithm(solver);

//     if (pbStopFlag)
//         optimizer.setForceStopFlag(pbStopFlag);

//     unsigned long maxKFid = 0;

//     // step7：添加顶点，Pose of Local KeyFrame
//     // cout << "(Optimizer) LocalBAwithLines - step7" << endl;
//     for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
//         KeyFrame* pKFi = *lit;
//         g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
//         vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
//         vSE3->setId(pKFi->mnId);
//         vSE3->setFixed(pKFi->mnId == 0);  // 第一个关键帧需要Fix
//         optimizer.addVertex(vSE3);
//         if (pKFi->mnId > maxKFid)
//             maxKFid = pKFi->mnId;
//     }

//     // step8:添加固定帧的顶点，Pose of Fixed KeyFrame
//     // cout << "(Optimizer) LocalBAwithLines - step8" << endl;
//     for (list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++) {
//         KeyFrame* pKFi = *lit;
//         g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
//         vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
//         vSE3->setId(pKFi->mnId);
//         vSE3->setFixed(true);
//         optimizer.addVertex(vSE3);
//         if (pKFi->mnId > maxKFid)
//             maxKFid = pKFi->mnId;
//     }

//     vector<int> MapPointID;

//     // step9：添加MapPoint的3D顶点
//     // cout << "(Optimizer) LocalBAwithLines - step9" << endl;
//     const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

//     vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
//     vpEdgesMono.reserve(nExpectedSize);

//     vector<KeyFrame*> vpEdgeKFMono;
//     vpEdgeKFMono.reserve(nExpectedSize);

//     vector<MapPoint*> vpMapPointEdgeMono;
//     vpMapPointEdgeMono.reserve(nExpectedSize);

//     const float thHuberMono = sqrt(5.991);

//     for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
//         MapPoint* pMP = *lit;
//         g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
//         vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
//         int id = pMP->mnId + maxKFid + 1;
//         vPoint->setId(id);
//         vPoint->setMarginalized(true);
//         optimizer.addVertex(vPoint);
//         MapPointID.push_back(id);

//         const map<KeyFrame*, size_t > observations = pMP->GetObservations();

//         // Set Edges
//         for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
//             KeyFrame* pKFi = mit->first;
//             if (!pKFi->isBad()) {
//                 const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

//                 // Monocular observation
//                 Eigen::Matrix<double, 2, 1> obs;
//                 obs << kpUn.pt.x, kpUn.pt.y;

//                 g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

//                 e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
//                 e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
//                 e->setMeasurement(obs);
//                 const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
//                 e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

//                 g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                 e->setRobustKernel(rk);
//                 rk->setDelta(thHuberMono);

//                 e->fx = pKFi->fx;
//                 e->fy = pKFi->fy;
//                 e->cx = pKFi->cx;
//                 e->cy = pKFi->cy;

//                 optimizer.addEdge(e);
//                 vpEdgesMono.push_back(e);
//                 vpEdgeKFMono.push_back(pKFi);
//                 vpMapPointEdgeMono.push_back(pMP);
//             }
//         }
//     }

//     sort(MapPointID.begin(), MapPointID.end());
//     int maxMapPointID = MapPointID[MapPointID.size() - 1];

//     // step10：添加MapLine的顶点
//     // cout << "(Optimizer) LocalBAwithLines - step10" << endl;
//     const int nLineExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapLines.size();

//     vector<EdgeLineProjectXYZ*> vpLineEdgesSP;
//     vpLineEdgesSP.reserve(nLineExpectedSize);

//     vector<EdgeLineProjectXYZ*> vpLineEdgesEP;
//     vpLineEdgesEP.reserve(nLineExpectedSize);

//     vector<KeyFrame*> vpLineEdgeKF;
//     vpLineEdgeKF.reserve(nLineExpectedSize);

//     vector<MapLine*> vpMapLineEdge;
//     vpMapLineEdge.reserve(nLineExpectedSize);
    
//     cout << "(Optimizer) LocalMapLines size: " << lLocalMapLines.size() << endl;
//     vector<int> MapLineSPID;
//     // 起始点的顶点
//     for (list<MapLine*>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
//         MapLine* pML = *lit;
//         g2o::VertexSBAPointXYZ* vStartP = new g2o::VertexSBAPointXYZ();
//         vStartP->setEstimate(pML->GetWorldPos().head(3));
//         int ids = pML->mnId + maxKFid + maxMapPointID + 1;
//         vStartP->setId(ids);
//         vStartP->setMarginalized(true);
//         optimizer.addVertex(vStartP);
//         MapLineSPID.push_back(ids);

//         const map<KeyFrame*, size_t > observations = pML->GetObservations();

//         for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
//             KeyFrame* pKFi = mit->first;
//             if (!pKFi->isBad()) {
//                 Eigen::Vector3d line_obs;
//                 line_obs = pKFi->mvKeyLineFunctions[mit->second];

//                 EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

//                 e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ids)));
//                 e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
//                 e->setMeasurement(line_obs);
//                 e->setInformation(Eigen::Matrix3d::Identity() * invSigma);

//                 g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
//                 e->setRobustKernel(rk_line_s);
//                 rk_line_s->setDelta(thHuberMono);

//                 e->fx = pKFi->fx;
//                 e->fy = pKFi->fy;
//                 e->cx = pKFi->cx;
//                 e->cy = pKFi->cy;

//                 e->Xw = pML->mWorldPos.head(3);

//                 optimizer.addEdge(e);
//                 vpLineEdgesSP.push_back(e);
//                 vpLineEdgeKF.push_back(pKFi);
//                 vpMapLineEdge.push_back(pML);
//             }
//         }
//     }

//     sort(MapLineSPID.begin(), MapLineSPID.end());
//     cout << "(Optimizer) MapLineSPID size: " << MapLineSPID.size() << endl;
//     int maxMapLineSPID = MapLineSPID[MapLineSPID.size() - 1];
//     // 终止点的顶点
//     for (list<MapLine*>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
//         MapLine* pML = *lit;
//         g2o::VertexSBAPointXYZ* vEndP = new VertexSBAPointXYZ();
//         vEndP->setEstimate(pML->GetWorldPos().tail(3));
//         int ide = pML->mnId + maxKFid + maxMapPointID + maxMapLineSPID + 1;
//         vEndP->setId(ide);
//         vEndP->setMarginalized(true);
//         optimizer.addVertex(vEndP);

//         const map<KeyFrame*, size_t> observations = pML->GetObservations();

//         for (map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
//             KeyFrame* pKFi = mit->first;
//             if (!pKFi->isBad()) {
//                 Eigen::Vector3d line_obs;
//                 line_obs = pKFi->mvKeyLineFunctions[mit->second];

//                 EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

//                 e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ide)));
//                 e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
//                 e->setMeasurement(line_obs);
//                 e->setInformation(Eigen::Matrix3d::Identity() * invSigma);

//                 g2o::RobustKernelHuber* rk_line_e = new g2o::RobustKernelHuber;
//                 e->setRobustKernel(rk_line_e);
//                 rk_line_e->setDelta(thHuberMono);

//                 e->fx = pKF->fx;
//                 e->fy = pKF->fy;
//                 e->cx = pKF->cx;
//                 e->cy = pKF->cy;

//                 e->Xw = pML->mWorldPos.head(3);

//                 optimizer.addEdge(e);
//                 vpLineEdgesEP.push_back(e);
//                 vpLineEdgeKF.push_back(pKFi);
//             }
//         }
//     }

//     if (pbStopFlag)
//         if (*pbStopFlag)
//             return;

//     // cout << "(Optimizer) LocalBAwithLines - optimize" << endl;
//     optimizer.initializeOptimization();
//     optimizer.optimize(5);

//     bool bDoMore = true;

//     if (pbStopFlag)
//         if (*pbStopFlag)
//             bDoMore = false;

//     if( bDoMore) {
//         // Check inlier observations
//         for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
//             g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
//             MapPoint* pMP = vpMapPointEdgeMono[i];

//             if (pMP->isBad())
//                 continue;

//             if (e->chi2() > 5.991 || !e->isDepthPositive()) {
//                 e->setLevel(1);
//             }
//             e->setRobustKernel(0);
//         }

//         for (size_t i = 0, iend = vpLineEdgesSP.size(); i < iend; i++) {
//             EdgeLineProjectXYZ* e = vpLineEdgesSP[i];
//             MapLine* pML = vpMapLineEdge[i];

//             if (pML->isBad())
//                 continue;

//             if (e->chi2() > 7.815) {
//                 e->setLevel(1);
//             }
//             e->setRobustKernel(0);
//         }

//         for (size_t i = 0, iend = vpLineEdgesEP.size(); i < iend; i++) {
//             EdgeLineProjectXYZ* e = vpLineEdgesEP[i];
//             MapLine* pML = vpMapLineEdge[i];

//             if (pML->isBad())
//                 continue;

//             if (e->chi2() > 7.815) {
//                 e->setLevel(1);
//             }
//             e->setRobustKernel(0);
//         }

//         // Optimize again without the outliers
//         optimizer.initializeOptimization(0);
//         optimizer.optimize(10);
//     }

//     vector<pair<KeyFrame*, MapPoint*>> vToErase;
//     vToErase.reserve(vpEdgesMono.size());

//     // check inlier observations
//     // cout << "(Optimizer) LocalBAwithLines - check inlier observations" << endl;
//     for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
//         g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
//         MapPoint* pMP = vpMapPointEdgeMono[i];

//         if (pMP->isBad())
//             continue;

//         if (e->chi2() > 5.991 || !e->isDepthPositive()) {
//             KeyFrame* pKFi = vpEdgeKFMono[i];
//             vToErase.push_back(make_pair(pKFi, pMP));
//         }
//     }
    
//     vector<pair<KeyFrame*, MapLine*>> vLineToErase;
//     vLineToErase.reserve(vpLineEdgesSP.size());
//     for (size_t i = 0, iend = vpLineEdgesSP.size(); i < iend; i++) {
//         EdgeLineProjectXYZ* e = vpLineEdgesSP[i];
//         MapLine* pML = vpMapLineEdge[i];

//         if (pML->isBad())
//             continue;

//         if (e->chi2() > 7.815) {
//             KeyFrame* pKFi = vpLineEdgeKF[i];
//             vLineToErase.push_back(make_pair(pKFi, pML));
//         }
//     }

//     // Get Map Mutex
//     unique_lock<mutex> lock(pMap->mMutexMapUpdate);

//     if (!vToErase.empty()) {
//         for (size_t i = 0; i < vToErase.size(); i++) {
//             KeyFrame* pKFi = vToErase[i].first;
//             MapPoint* pMPi = vToErase[i].second;
//             pKFi->EraseMapPointMatch(pMPi);
//             pMPi->EraseObservation(pKFi);
//         }
//     }

//     if (!vLineToErase.empty()) {
//         for (size_t i = 0; i < vLineToErase.size(); i++) {
//             KeyFrame* pKFi = vLineToErase[i].first;
//             MapLine* pMLi = vLineToErase[i].second;
//             pKFi->EraseMapLineMatch(pMLi);
//             pMLi->EraseObservation(pKFi);
//         }
//     }

//     // Recover optimized data
//     // cout << "(Optimizer) LocalBAwithLines - recover optimized data" << endl;
//     // Keyframes
//     for (list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
//         KeyFrame* pKF = *lit;
//         g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
//         g2o::SE3Quat SE3quat = vSE3->estimate();
//         pKF->SetPose(Converter::toCvMat(SE3quat));
//     }

//     // Points
//     for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++) {
//         MapPoint* pMP = *lit;
//         g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
//         pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
//         pMP->UpdateNormalAndDepth();
//     }

//     // Lines
//     for (list<MapLine*>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
//         MapLine* pML = *lit;
//         g2o::VertexSBAPointXYZ* vStartP = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + 1));
//         g2o::VertexSBAPointXYZ* vEndP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + maxMapLineSPID + 1));

//         Vector6d LinePos;
//         LinePos << Converter::toVector3d(Converter::toCvMat(vStartP->estimate())), Converter::toVector3d(Converter::toCvMat(vEndP->estimate()));
//         pML->SetWorldPos(LinePos);
//         pML->UpdateAverageDir();
//     }
// }

void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale) {
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0, 2);
    vSim3->_principle_point1[1] = K1.at<float>(1, 2);
    vSim3->_focal_length1[0] = K1.at<float>(0, 0);
    vSim3->_focal_length1[1] = K1.at<float>(1, 1);
    vSim3->_principle_point2[0] = K2.at<float>(0, 2);
    vSim3->_principle_point2[1] = K2.at<float>(1, 2);
    vSim3->_focal_length2[0] = K2.at<float>(0, 0);
    vSim3->_focal_length2[1] = K2.at<float>(1, 1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for (int i = 0; i < N; i++) {
        if (!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if (pMP1 && pMP2) {
            if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0) {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double, 2, 1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++) {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if (nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;

    if (nCorrespondences - nBad < 10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++) {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
