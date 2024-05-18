#include <pangolin/pangolin.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>


// used for pangolin
#define POINT_SIZE  1.5
#define LINE_WIDTH  2.5

#define VIEW_POINT_X  0
#define VIEW_POINT_Y  -0.7
#define VIEW_POINT_Z  -1.8
#define VIEW_POINT_F  500


class PointFeature3D {
public:
    float xw;
    float yw;
    float zw;

public:
    PointFeature3D(const float& _xw, const float& _yw, const float& _zw): xw(_xw), yw(_yw), zw(_zw) {}
};


class LineFeature3D {
public:
    float sx, sy, sz;
    float ex, ey, ez;
    float r, g, b;
    std::string label;

public:
    LineFeature3D(
        const float& _sx, const float& _sy, const float& _sz,
        const float& _ex, const float& _ey, const float& _ez,
        const float& _r, const float& _g, const float& _b, 
        const std::string& _label
    ): sx(_sx), sy(_sy), sz(_sz), ex(_ex), ey(_ey), ez(_ez), r(_r), g(_g), b(_b), label(_label) {}
};


class PointFeature2D {
public:
    float uc;
    float vc;

public:
    PointFeature2D(const float& _uc, const float& _vc): uc(_uc), vc(_vc) {}
};


class LineFeature2D {
public:
    float su, sv;
    float eu, ev;
    int r, g, b;
    std::string label;

public:
    LineFeature2D(
        const float& _su, const float& _sv,
        const float& _eu, const float& _ev,
        const int& _r, const int& _g, const int& _b, 
        const std::string& _label
    ): su(_su), sv(_sv), eu(_eu), ev(_ev), r(_r), g(_g), b(_b), label(_label) {}
};


void load3DCloudFile(
        const std::string filename, 
        std::vector<PointFeature3D>& pointFeatures, std::vector<LineFeature3D>& lineFeatures
    ) {
    std::cout << "Loading point cloud file..." << std::endl << std::endl;
    std::ifstream fCloud;
    fCloud.open(filename.c_str());
    while (!fCloud.eof()) {
        std::string s;
        std::getline(fCloud, s);
        if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            std::string feature;
            ss >> feature;
            if ('#' == feature[0]) {
                continue;
            } else if ("point" == feature) {
                double xw, yw, zw;
                ss >> xw;
                ss >> yw;
                ss >> zw;
                pointFeatures.push_back(PointFeature3D(xw, yw, zw));
            } else if ("line" == feature) {
                double sx, sy, sz, ex, ey, ez, r, g, b;
                std::string label;
                ss >> sx;  ss >> sy;  ss >> sz;
                ss >> ex;  ss >> ey;  ss >> ez;
                ss >> r;  ss >> g;  ss >> b;  ss >> label;
                lineFeatures.push_back(LineFeature3D(sx, sy, sz, ex, ey, ez, r, g, b, label));
            }
        }
    }
    std::cout << "Get point features num: " << pointFeatures.size() << std::endl;
    std::cout << "Get line features num: " << lineFeatures.size() << std::endl;
    std::cout << "Point cloud loaded." << std::endl << std::endl;
}


void load2DCloudFile(
        const std::string filename, 
        std::vector<PointFeature2D>& pointFeatures, std::vector<LineFeature2D>& lineFeatures
    ) {
    std::cout << "Loading point cloud file..." << std::endl << std::endl;
    std::ifstream fCloud;
    fCloud.open(filename.c_str());
    while (!fCloud.eof()) {
        std::string s;
        std::getline(fCloud, s);
        if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            std::string feature;
            ss >> feature;
            if ('#' == feature[0]) {
                continue;
            } else if ("point" == feature) {
                double uc, vc;
                ss >> uc;
                ss >> vc;
                pointFeatures.push_back(PointFeature2D(uc, vc));
            } else if ("line" == feature) {
                double su, sv, eu, ev, r, g, b;
                std::string label;
                ss >> su;  ss >> sv;
                ss >> eu;  ss >> ev;
                ss >> r;  ss >> g;  ss >> b;  ss >> label;
                lineFeatures.push_back(LineFeature2D(
                    su, sv, eu, ev, 
                    int(r * 255.0 + 0.5), int(g * 255.0 + 0.5), int(b * 255.0 + 0.5), 
                    label
                ));
            }
        }
    }
    std::cout << "Get point features num: " << pointFeatures.size() << std::endl;
    std::cout << "Get line features num: " << lineFeatures.size() << std::endl;
    std::cout << "Point cloud loaded." << std::endl << std::endl;
}


void drawPangolinPoints(const std::vector<PointFeature3D>& pointFeatures) {
    glPointSize(POINT_SIZE);
    glBegin(GL_POINT);
    glColor3f(0.0, 0.0, 0.0);  // black
    for (auto& point: pointFeatures) {
        glVertex3f(point.xw, point.yw, point.zw);
    }
    glEnd();
}


void drawPangolinLines(const std::vector<LineFeature3D>& lineFeatures) {
    std::map<std::string, std::vector<LineFeature3D>> lineMap;
    for (auto& line: lineFeatures) {
        lineMap[line.label].push_back(line);
    }

    std::map<std::string, std::vector<LineFeature3D>>::iterator iter;
    for (iter = lineMap.begin(); iter != lineMap.end(); iter++) {
        glLineWidth(LINE_WIDTH);
        glBegin(GL_LINES);
        glColor3f(iter->second[0].r, iter->second[0].g, iter->second[0].b);
        for (auto& line: iter->second) {
            glVertex3f(line.sx, line.sy, line.sz);
            glVertex3f(line.ex, line.ey, line.ez);
        }
        glEnd();
    }
}


void drawWithPangolin(
        std::vector<PointFeature3D>& pointFeatures, std::vector<LineFeature3D>& lineFeatures
    ) {
    pangolin::CreateWindowAndBind("Sparse Point Cloud", 1024, 768);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, VIEW_POINT_F, VIEW_POINT_F, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(VIEW_POINT_X, VIEW_POINT_Y, VIEW_POINT_Z, 0, 0, 0, 0.0, -1.0, 0.0)
    );
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        drawPangolinPoints(pointFeatures);
        drawPangolinLines(lineFeatures);

        pangolin::FinishFrame();
    }
}


void drawWithOpenCV(
        std::vector<PointFeature2D>& pointFeatures, std::vector<LineFeature2D>& lineFeatures
    ) {
    cv::Mat background(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto& point: pointFeatures) {
        cv::Point2f pt(point.uc, point.vc);
        cv::circle(background, pt, 3, cv::Scalar(0, 0, 0), 1);
    }

    for (auto& line: lineFeatures) {
        cv::Point2f sp(line.su, line.sv), ep(line.eu, line.ev);
        cv::line(background, sp, ep, cv::Scalar(line.b, line.g, line.r), 2);
    }

    cv::imshow("Sparse Point Cloud", background);
    cv::waitKey(0);
}


void printCommandTips() {
    std::cerr << "Wrong arguments!" << std::endl;
    std::cout << "Command format 1: ./show_sparse_cloud 2d /PATH/TO/SPARSE/CLOUD/FILE" << std::endl;
    std::cout << "  - Format of file contents should be: " << std::endl;
    std::cout << "      - 1): point float(u_c) float(v_c)" << std::endl;
    std::cout << "      - 2): line float(s_u) float(s_v) float(e_u) float(e_v) float(r) float(g) float(b) string(label)" << std::endl;
    std::cout << std::endl;
    std::cout << "Command format 2: ./show_sparse_cloud 3d /PATH/TO/SPARSE/CLOUD/FILE" << std::endl;
    std::cout << "  - Format of file contents should be: " << std::endl;
    std::cout << "      - 1): point float(x_w) float(y_w)" << std::endl;
    std::cout << "      - 2): line float(s_x) float(s_y) float(s_z) float(e_x) float(e_y) float(e_z) float(r) float(g) float(b) string(label)" << std::endl;
}


int main(int argc, char **argv) {
    if (argc != 3) {
        printCommandTips();
        return 1;
    }

    std::string dim = std::string(argv[1]);
    std::string cloudFile = std::string(argv[2]);

    if ("3d" == dim) {
        std::vector<PointFeature3D> pointFeatures;
        std::vector<LineFeature3D> lineFeatures;
        load3DCloudFile(cloudFile, pointFeatures, lineFeatures);
        drawWithPangolin(pointFeatures, lineFeatures);
    } else if ("2d" == dim) {
        std::vector<PointFeature2D> pointFeatures;
        std::vector<LineFeature2D> lineFeatures;
        load2DCloudFile(cloudFile, pointFeatures, lineFeatures);
        drawWithOpenCV(pointFeatures, lineFeatures);
    } else {
        printCommandTips();
        return 1; 
    }

    return 0;
}
