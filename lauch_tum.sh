./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml YoloConfig /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_walking_halfsphere /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_walking_halfsphere/associate.txt
mv ./Output/CameraTrajectory.txt ./Output/pl-fr3-w-half-wo.txt
mv ./Output/KeyFrameTrajectory.txt ./Output/pl-fr3-w-half-wo-keyframe.txt
mv './Output/MapFeatures(Point-Line).txt' ./Output/pl-fr3-w-half-wo-sparsecloud.txt
mv ./Output/GlobalPointCloud.pcd ./Output/pl-fr3-w-half-wo-densecloud.pcd

./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml YoloConfig /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_walking_rpy /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_walking_rpy/associate.txt
mv ./Output/CameraTrajectory.txt ./Output/pl-fr3-w-rpy-wo.txt
mv ./Output/KeyFrameTrajectory.txt ./Output/pl-fr3-w-rpy-wo-keyframe.txt
mv './Output/MapFeatures(Point-Line).txt' ./Output/pl-fr3-w-rpy-wo-sparsecloud.txt
mv ./Output/GlobalPointCloud.pcd ./Output/pl-fr3-w-rpy-wo-densecloud.pcd

./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml YoloConfig /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_walking_static /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_walking_static/associate.txt
mv ./Output/CameraTrajectory.txt ./Output/pl-fr3-w-static-wo.txt
mv ./Output/KeyFrameTrajectory.txt ./Output/pl-fr3-w-static-wo-keyframe.txt
mv './Output/MapFeatures(Point-Line).txt' ./Output/pl-fr3-w-static-wo-sparsecloud.txt
mv ./Output/GlobalPointCloud.pcd ./Output/pl-fr3-w-static-wo-densecloud.pcd

./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml YoloConfig /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_walking_xyz /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_walking_xyz/associate.txt
mv ./Output/CameraTrajectory.txt ./Output/pl-fr3-w-xyz-wo.txt
mv ./Output/KeyFrameTrajectory.txt ./Output/pl-fr3-w-xyz-wo-keyframe.txt
mv './Output/MapFeatures(Point-Line).txt' ./Output/pl-fr3-w-xyz-wo-sparsecloud.txt
mv ./Output/GlobalPointCloud.pcd ./Output/pl-fr3-w-xyz-wo-densecloud.pcd


mv ./YoloConfig/yolov5_config.yaml ./YoloConfig/yolov5_config_wo.yaml
mv ./YoloConfig/yolov5_config_od.yaml ./YoloConfig/yolov5_config.yaml


./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml YoloConfig /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_sitting_halfsphere /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_sitting_halfsphere/associate.txt
mv ./Output/CameraTrajectory.txt ./Output/pl-fr3-s-half-wo.txt
mv ./Output/KeyFrameTrajectory.txt ./Output/pl-fr3-s-half-wo-keyframe.txt
mv './Output/MapFeatures(Point-Line).txt' ./Output/pl-fr3-s-half-wo-sparsecloud.txt
mv ./Output/GlobalPointCloud.pcd ./Output/pl-fr3-s-half-wo-densecloud.pcd

./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml YoloConfig /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_sitting_rpy /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_sitting_rpy/associate.txt
mv ./Output/CameraTrajectory.txt ./Output/pl-fr3-s-rpy-wo.txt
mv ./Output/KeyFrameTrajectory.txt ./Output/pl-fr3-s-rpy-wo-keyframe.txt
mv './Output/MapFeatures(Point-Line).txt' ./Output/pl-fr3-s-rpy-wo-sparsecloud.txt
mv ./Output/GlobalPointCloud.pcd ./Output/pl-fr3-s-rpy-wo-densecloud.pcd

./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml YoloConfig /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_sitting_static /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_sitting_static/associate.txt
mv ./Output/CameraTrajectory.txt ./Output/pl-fr3-s-static-wo.txt
mv ./Output/KeyFrameTrajectory.txt ./Output/pl-fr3-s-static-wo-keyframe.txt
mv './Output/MapFeatures(Point-Line).txt' ./Output/pl-fr3-s-static-wo-sparsecloud.txt
mv ./Output/GlobalPointCloud.pcd ./Output/pl-fr3-s-static-wo-densecloud.pcd

./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml YoloConfig /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_sitting_xyz /mnt/e/TUM/DynamicObjects/rgbd_dataset_freiburg3_sitting_xyz/associate.txt
mv ./Output/CameraTrajectory.txt ./Output/pl-fr3-s-xyz-wo.txt
mv ./Output/KeyFrameTrajectory.txt ./Output/pl-fr3-s-xyz-wo-keyframe.txt
mv './Output/MapFeatures(Point-Line).txt' ./Output/pl-fr3-s-xyz-wo-sparsecloud.txt
mv ./Output/GlobalPointCloud.pcd ./Output/pl-fr3-s-xyz-wo-densecloud.pcd


mv ./YoloConfig/yolov5_config.yaml ./YoloConfig/yolov5_config_od.yaml
mv ./YoloConfig/yolov5_config_wo.yaml ./YoloConfig/yolov5_config.yaml
