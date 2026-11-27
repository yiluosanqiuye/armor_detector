#include "pose.h"

#include <vector>
#include <cmath>

using namespace std;
using namespace cv;


// 装甲板模型尺寸（单位：米）
static const double PLATE_W = 0.135;
static const double PLATE_H = 0.056;


// 四个模型点顺序
static const vector<Point3d> OBJECT_POINTS = {
    //模型点顺序必须与 Armor::corners一一对应,顺序错一个，解出的位姿会是错的，但重投影误差可能仍然很小
    {-PLATE_W / 2, -PLATE_H / 2, 0},   // 0-左上
    { PLATE_W / 2, -PLATE_H / 2, 0},   // 1-右上
    { PLATE_W / 2,  PLATE_H / 2, 0},   // 2-右下
    {-PLATE_W / 2,  PLATE_H / 2, 0}    // 3-左下

};


// 相机内参-K
static const Mat K = (Mat_<double>(3, 3) <<
    1781.053454, 0, 510.5751561,
    0, 1781.686873, 472.9735763,
    0, 0, 1);


// 畸变系数-DIST
static const Mat DIST = (Mat_<double>(1, 5) <<
    -0.04420740074, -0.1081537096,
    -0.0006506219039, 0.0002357380215, 0.466737247);


// 获取位姿
Pose estimate_pose(const array<Point2f, 4> &corners) {
    Pose pose;

    //1.转换装甲板角点类型以便pnp解算
    vector<Point2d> plate_image_points;
    for (int i = 0; i < 4; i++) {
        plate_image_points.emplace_back(corners[i]);
    }


    //2.solvePnP-解位姿
    Mat rvec,tvec;
    bool pnp_ok = solvePnP(OBJECT_POINTS, plate_image_points, K, DIST, rvec, tvec);


    // 3.确认位姿的有效性
    if (pnp_ok==false) {
        return pose;                        // solvePnP解算失败，valid保持false
    }

    for (int i = 0; i < 3; i++) {
        if (!isfinite(tvec.at<double>(i)) || !isfinite(rvec.at<double>(i))) {
            return pose;                    // 非有限数值结果，valid保持false
        }
    }

    if (tvec.at<double>(2) <= 0) {
        return pose;                        // 深度为负，valid保持false
    }


    //4.记录位姿信息
    pose.valid    = true;
    pose.tvec     = Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    pose.rvec     = Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    pose.distance = norm(pose.tvec);


    //5.重投影
    vector<Point2d> plate_reproject_points;
    projectPoints(OBJECT_POINTS, rvec, tvec, K, DIST, plate_reproject_points);

    double sum_error = 0.0;
    for (int i = 0; i < 4; i++) {
        pose.reproject_corners[i] = plate_reproject_points[i];
        sum_error += norm(plate_reproject_points[i] - plate_image_points[i]);
    }
    pose.reproject_error = sum_error / 4.0;          //算四个角点平均像素误差



    return pose;
}