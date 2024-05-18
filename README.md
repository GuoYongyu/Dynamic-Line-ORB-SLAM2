# Dynamic-Line-ORB-SLAM2

A ORB-SLAM2 instance with dynamic object removing and point-line features optimization.

This project is an experimental improvement based on ORB-SLAM2:
- The extraction of object semantic masks is completed based on YOLOv5 and depth images. This method is derived from the paper Dynamic-VINS, but has been modified to use sampling and RANSAC fitting instead of the original method that used the depth of the detection box center point as the object depth.
- On the basis of ORB-SLAM2, LSD line features are introduced and incorporated into pose optimization.
- The system output two types of maps (Sparse Point-Line Features Map saved as `.txt` file, and Dense RGB Pointcloud Map saved as `.pcd` file).

# 1. Prerequisites

This project is developed based on `Ubuntu-18.04`. In theory, it can run on other Ubuntu systems such as `16.04`, `20.04`.

## 1.1. C++11
Project uses the new thread and chrono functionalities of C++11.

## 1.2. Pangolin
Project uses Pangolin for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## 1.3. Eigen
Eigen's version here is `3.3.4`, which is needed by g2o. Perhaps version `3.1.0` is available too, but project did not try when delevoping.

## 1.4. OpenCV4
OpenCV is mainly used to process images and YOLOv5 in this project. Project uses `dnn` module in OpenCV to load YOLOv5, therefore, you need to download and install `OpenCV4`. In experiments, project used version `4.5`. You can download OpenCV4 at http://opencv.org.

In new OpenCV version, line segments detection is removed. There is only one `lsd.cpp` file in OpenCV, and the functions inside cannot be called or cannot obtain results after calling. We need to change lsd.cc first:
- Download old OpenCV, such as OpenCV-3.1.0, copy `lsd.cpp` under `modules/imgproc/src` and replace the same name file at the same folder in `OpenCV-4.5.X`, then start to compile OpenCV.
- You can refer to: https://blog.csdn.net/Zhaoxi_Li/article/details/106846313

Here, we need to compile OpenCV4 with GPU, because we want to use GPU to process the YOLOv5. Of course, compile without GPU is okay, but the image processing will be slow. If compile with GPU, the command can be:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_TBB=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_CUDA=ON \
      -D BUILD_opencv_cudacodec=OFF \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_BIN=8.6 \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D WITH_GSTREAMER=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_PC_FILE_NAME=opencv.pc \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/cv/lib/python3.8/site-packages \
      -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
      -D OPENCV_EXTRA_MODULES_PATH=/home/dxc/opencv_contrib-4.5.4/modules \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D BUILD_EXAMPLES=OFF ..
```
You can refer to: https://zhuanlan.zhihu.com/p/685513944
or: https://blog.csdn.net/weixin_43850132/article/details/130367507

## 1.5. DBoW2 and g2o
Both two libraries are used in ORB-SLAM2, but g2o was modified by ORB-SLAM2. Two libraries are compressed and updated at `Thirdparty`.

# 2. Building

## 2.1. Download

### 2.1.1. Vocabulary

Download vocabulary file from orginal ORB-SLAM2 project: https://github.com/raulmur/ORB_SLAM2
or download from Google Drive: https://drive.google.com/file/d/1B4rDUUmaCqAZ9J9npuNivWu10IYNw2dP/view?usp=drive_link

Then, move the zip file to folder `Vocabulary` (You can unzip the file first, or the file will be unzipped automatically when building.)

### 2.1.2. YOLOv5

YOLOv5's home: https://github.com/ultralytics/yolov5

We need to download pre-trained model parameters from YOLOv5: https://github.com/ultralytics/yolov5/releases
The YOLOv5 version used in project is `v6.0`, other versions are available. Also, you can train your own model with other datasets (YOLOv5 used COCO). If you do like this, you need to change some configurations in some files under folder `YoloConfig`. Because new labels will affects some configuration files mainly `coco.names` at `YoloConfig/models/yolo/coco.names`.

Due to using OpenCV to load YOLOv5, we need to transfer parameter files first. The transfered parameter files (YOLOv5-v6.0) can be downloaded from: https://drive.google.com/drive/folders/1iOTPI8JTHtWq36eB1UIR45s_s8N9qZuY?usp=drive_link
You can refer to the tutorial to convert to other versions: https://blog.csdn.net/weixin_41311686/article/details/128421801

## 2.2. Build

Clone the repository:
```
git clone https://github.com/GuoYongyu/Dynamic-Line-ORB-SLAM2.git DL_ORB_SLAM2
```

Run "./build.sh" at home directory to build the project:
```
cd DL_ORB_SLAM2
sudo chmod +x build.sh
./build.sh
```

This will create libORB_SLAM2.so at `lib` folder and the executables `rgbd_tum`, `show_sparse_cloud` in Examples folder.

# 3. Run

- Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.
- Associate RGB images and depth images using the python script associate.py. We already provide associations for some of the sequences in Examples/RGB-D/associations/. You can generate your own associations file executing (`associate.py` is provided at home folder):
```
python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```
- Create a new folder named `Output` at home folder.
- Execute the following command. Change TUMX.yaml to TUM1.yaml, TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change PATH_TO_SEQUENCE_FOLDER to the uncompressed sequence folder:
```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml YoloConfig PATH_TO_SEQUENCE_FOLDER
```
- Use command below and follow the tips to show sparse map:
```
./Examples/RGB-D/show_sparse_cloud
```

# 4. Details

- The semantic masks detection is referenced from Dynamic-VINS: https://github.com/HITSZ-NRSL/Dynamic-VINS, we modified the algorithm with RANSAC.
- The line features process is referenced from: https://github.com/atlas-jj/ORB_Line_SLAM. This project only process line in Monocular SLAM, project refers to the methods used and transplant them to RGB-D SLAM.
- The pangolin project used in our PC may be damaged, point & line features cannot be show properly, but the output is normal.
