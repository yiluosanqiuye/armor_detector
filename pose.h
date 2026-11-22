#pragma once

#include <opencv2/opencv.hpp>
#include <array>


//一块装甲板的位姿结果
struct Pose {
    bool       valid           = false;
    cv::Vec3d  tvec{0, 0, 0};
    cv::Vec3d  rvec{0, 0, 0};
    double     distance        = 0.0;     //相机光心到装甲中心的三维斜距
    double     reproject_error = 0.0;
};

//输入四个角点，返回位姿
Pose estimate_pose(const std::array<cv::Point2f, 4> &corners);