#include "video.h"
#include "detector.h"
#include "pose.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace cv;


// 视频基本属性
struct Video {
    int    width       = 0;
    int    height      = 0;
    double fps         = 0.0;
    int    frame_count = 0;
};


// 位姿统计（跨帧累计）
struct PoseStats {
    int    ok       = 0;        //解算成功次数
    int    fail     = 0;        //解算失败次数
    double tz_min   = 1e9;
    double tz_max   = -1e9;
    double tz_sum   = 0.0;
    int    tz_count = 0;

    //统计数据
    void add(const Pose &p) {
        if (p.valid==false) {
            fail++;
        }
        else {
            ok++;
            double tz = p.tvec[2];
            tz_min = min(tz_min, tz);
            tz_max = max(tz_max, tz);
            tz_sum += tz;
            tz_count++;
        }
    }
};


// 候选对象（装甲板+对应位姿）
struct Candidate {
    int id;
    Armor armor;
    Pose  pose;

    Candidate(){};
    Candidate(int i,Armor a,Pose p):id(i),armor(a),pose(p){};
};


// 装甲板组排序
bool armor_center_cmp(const Armor &a,const Armor &b) {
    //中心点从左到右排序
    if ( a.center.x != b.center.x) {
        return a.center.x < b.center.x;
    }
    else {
        return a.center.y < b.center.y;
    }
}

//-----------------------------------------主流程函数----------------------------------------------


// 打开视频
static bool open_video(VideoCapture &cap, const string &path) {
    cap.open(path, CAP_FFMPEG);

    if (!cap.isOpened()) {
        cerr << "FFmpeg 后端打不开，尝试默认后端..." << endl;
        cap.open(path);

        if (!cap.isOpened()) {
            cerr << "视频打开失败：" << path << endl;

            //还是打开失败，那就打印当前OpenCV支持的视频后端，方便排查
            string information = getBuildInformation();
            size_t position = information.find("Video I/O");

            if (position != string::npos) {
                cerr << endl;
                cerr << "当前OpenCV的视频后端：" << endl;
                cerr << information.substr(position, 200) << endl;
            }
            return false;
        }
    }
    return true;
}


// 读取视频属性
static Video read_video_information(VideoCapture &cap) {
    Video video;

    video.width       = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    video.height      = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    video.fps         =      cap.get(CAP_PROP_FPS);
    video.frame_count = (int)cap.get(CAP_PROP_FRAME_COUNT);

    //读不到时回退到默认值
    if (video.width <= 0 || video.height <= 0) {
        video.width  = 1280;
        video.height = 1024;
        cerr << "警告：宽高读取失败，回退 1280x1024" << endl;
    }
    if (video.fps <= 0) {
        video.fps = 120.0;
        cerr << "警告：fps 读取失败，回退 120" << endl;
    }
    return video;
}


// 打印视频属性
static void print_video_information(const string &path, const Video &video) {
    cout << left << "视频路径：" << path << endl;
    cout << "----------------------------视频属性--------------------------------" << endl;
    cout << setw(18) << "width"       << video.width       << endl;
    cout << setw(18) << "height"      << video.height      << endl;
    cout << setw(18) << "fps"         << video.fps         << endl;
    cout << setw(18) << "frame_count" << video.frame_count << endl;
    cout << right << endl;
}


// 在图上标注候选板编号
static void draw_candidate_id(Mat &canva, const Armor &armor, int candidate_id) {
    putText(canva,
            "#" + to_string(candidate_id),
            Point(armor.center.x - 8, armor.corners[0].y - 8),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
}


// 在图上标注位姿信息
static void draw_information_panel(Mat &canva, int frame_id, const Video &video,
                                  const vector<Candidate> &candidates) {

    const int x      = 16;      // 左边距
    const int y0     = 34;      // 第一行字的纵坐标
    const int line_h = 26;      // 行距

    int y = y0;                 // 当前行的纵坐标

    //第一行————帧号和时间
    {
        ostringstream oss;
        oss << "frame " << frame_id << "    t = "
            << fixed << setprecision(2) << (frame_id / video.fps) << " s";

        putText(canva, oss.str(), Point(x, y),
                FONT_HERSHEY_SIMPLEX, 0.55, Scalar(230, 230, 230), 1, LINE_AA);

        y += line_h;
    }

    //第二行————候选数量
    putText(canva,
            "candidates: " + to_string((int)candidates.size()),
            Point(x, y),
            FONT_HERSHEY_SIMPLEX, 0.55, Scalar(230, 230, 230), 1, LINE_AA);

    y += line_h;

    //第三行————编号+状态+距离
    for (const auto& c:candidates) {
        const Armor &armor = c.armor;
        const Pose  &pose  = c.pose;

        string text;
        Scalar color;

        //检测失败（四角点不可用）
        if (armor.valid == false) {
            text  = "#" + to_string(c.id) + "   DET FAIL";
            color = Scalar(0, 0, 255);
        }
        //检测和位姿解算都成功
        else if (pose.valid == true) {
            ostringstream oss;
            oss << "#" << c.id << "   PnP OK   distance = "
                << fixed << setprecision(2) << pose.distance << " m";
            text  = oss.str();
            color = Scalar(0, 255, 0);
        }
        //检测成功但位姿解算失败
        else {
            text  = "#" + to_string(c.id) + "   PnP FAIL";
            color = Scalar(0, 0, 255);
        }

        putText(canva, text, Point(x, y),
                FONT_HERSHEY_SIMPLEX, 0.55, color, 1, LINE_AA);

        y += line_h;
    }
}


// 重投影误差
static void draw_reproject_compare(Mat &canva,const Candidate &c){

    //原图像角点用红色实心点显示，重投影角点用绿色空心圈显示
    for (int i=0;i<4;i++) {
        circle(canva,c.armor.corners[i],2,Scalar(0,0,255),-1);
        circle(canva,c.pose.reproject_corners[i],5,Scalar(0,255,0),1);
    }

}


// 粗略深度核对
static void depth_comparison(int frame_id,const Candidate &c) {

    double top_corners_px    =norm(c.armor.corners[0]-c.armor.corners[1]);
    double bottom_corners_px =norm(c.armor.corners[3]-c.armor.corners[2]);
    double armor_width_px    =(top_corners_px+bottom_corners_px)/2;
    const double rough_tz_m  = 1781.053454 * 0.135 / armor_width_px;

    cout << "---------------------粗略深度核对---------------------------" << endl;
    cout << left << "帧号-" << frame_id << '\n'
         << setw(18) << "armor_width_px" << armor_width_px << '\n'
         << setw(18) << "rough_tz_m"     << rough_tz_m     << '\n'
         << setw(18) << "pnp_tz_m"       << c.pose.tvec[2] << '\n'
         << right;
    cout << endl;
}


// 敏感性实验
static void pose_sensitivity_test(int frame_id,const Candidate &c) {
    // 定义扰动后的角点组
    auto perturbed_corners = c.armor.corners;
    perturbed_corners[0].x += 2.0f;

    const Pose& original_pose   = c.pose;
    const Pose  perturbed_pose = estimate_pose(perturbed_corners);

    if (perturbed_pose.valid==false) {
        cerr <<  "敏感性实验：角点0向右偏移2px 后，未得到有效位姿，跳过本次对比。" << endl;
        return;
    }

    cout << endl;
    cout << "-------------------敏感性实验-------------------------" << endl;
    cout << "帧号-" << frame_id << endl;
    cout << left
         << setw(28) << "tz (m)" << setw(8) << original_pose.tvec[2]
         << " -> " << setw(18) << perturbed_pose.tvec[2]
         << "变化 " << perturbed_pose.tvec[2] - original_pose.tvec[2] << '\n'
         << setw(28) << "distance (m)" << setw(8) << original_pose.distance
         << " -> " << setw(18) << perturbed_pose.distance
         << "变化 " << perturbed_pose.distance - original_pose.distance << '\n'
         << setw(28) << "reproject_error (px)" << setw(8) << original_pose.reproject_error
         << " -> " << setw(18) << perturbed_pose.reproject_error
         << "变化 "  << perturbed_pose.reproject_error - original_pose.reproject_error << '\n';
    cout << right << endl;

}


//处理单帧图像，并返回这一帧的候选装甲板数量
static int process_frame(Mat &frame, int frame_id, const Video &video,ofstream &csv, PoseStats &stats, Mat &canva) {
//每帧

    //检测
    vector<Armor> armors = detect_armors(frame);
    sort(armors.begin(),armors.end(),armor_center_cmp);       // 将装甲板按中心点位置排序，以便编号

    canva = frame.clone();

    vector<Candidate> candidates;

    //没有候选————写一行空记录
    if (armors.empty()) {
        csv << frame_id << "," << (frame_id / video.fps) << ",,0,0,,,,,,,," << endl;
        draw_information_panel(canva, frame_id, video, candidates);
        return 0;
    }

    Photo photo;

    for (const Armor &armor : armors) {
    //每个装甲板

        //检测装甲板是否有效（四角点是否可用）
        int detect_ok = armor.valid ? 1 : 0;

        //只有检测有效才解算位姿 （位姿有效必定装甲板有效，但装甲板有效不一定位姿有效）
        Pose pose;
        if (detect_ok == 1) {
            pose = estimate_pose(armor.corners);
            //位姿统计
            stats.add(pose);
        }

        //打包成候选（只要有候选装甲板就连同位姿存入候选（无论装甲板有效与否））
        Candidate candidate((int)candidates.size(),armor,pose);
        candidates.push_back(candidate);

        //检测重投影误差
        if (pose.valid==true&&pose.reproject_error>0.50) {
            cerr << "警告：" << "帧号-" << frame_id << "  candidate-#" << candidate.id
            << "重投影误差" << pose.reproject_error << "px，超过 0.50 px，请复核角点。" << endl;
        }
        //画框+标号
        photo.draw_armor(canva, armor);
        draw_candidate_id(canva, armor, candidate.id);

        //写CSV
        csv << frame_id << "," << (frame_id / video.fps) << ","
            << candidate.id << "," << detect_ok << ",";

        if (detect_ok == true && pose.valid == true) {
            csv << "1,"
                << pose.tvec[0] << "," << pose.tvec[1] << "," << pose.tvec[2] << ","
                << pose.distance << ","
                << pose.rvec[0] << "," << pose.rvec[1] << "," << pose.rvec[2] << ","
                << pose.reproject_error
                << endl;
        }
        else {
            csv << "0,,,,,,,," << endl;
        }

    }

    // 画信息面板
    draw_information_panel(canva, frame_id, video, candidates);

    // 检测重投影误差和粗略深度核对
    if (frame_id%333==0) {
        Mat reproject_canva=canva.clone();
        bool have_valid_pose=false;

        for (const auto& c:candidates) {
            if (c.pose.valid==false) {
                continue;
            }
            have_valid_pose=true;
            // 位姿有效才操作
            draw_reproject_compare(reproject_canva,c);
            depth_comparison(frame_id,c);
        }
        // 保存重投影误差效果图
        if (have_valid_pose==true) {
            imwrite("output/reproject_"+to_string(frame_id)+".png",reproject_canva);
        }
    }

    // 敏感性实验
    if (frame_id == 0 || frame_id == 2331 || frame_id == 3330) {
        for (const auto &c:candidates) {
            if (c.pose.valid==false) {
                continue;
            }
            pose_sensitivity_test(frame_id,c);
        }
    }

    return (int)candidates.size();
}


// ---------------------------------------视频主流程------------------------------------------------
int run_video(const string &path) {

    // 1.打开视频
    VideoCapture cap;
    if (!open_video(cap, path)) {
        return 1;
    }


    // 2.读取并打印属性
    Video video = read_video_information(cap);
    print_video_information(path, video);


    // 3.创建输出视频
    string out_path = "output/result.mp4";
    VideoWriter video_writer(out_path,
                             VideoWriter::fourcc('m', 'p', '4', 'v'),
                             video.fps,
                             Size(video.width, video.height));
    
    if (!video_writer.isOpened()) {
        cerr << "输出视频创建失败：" << out_path << endl;
        return 1;
    }
    cout << "输出视频：" << out_path << endl;
    cout << endl;


    // 4.打开逐帧数据文件csv
    ofstream csv("output/pose_log.csv");
    if (!csv.is_open()) {
        cerr << "无法创建 output/pose_log.csv" << endl;
        return 1;
    }

    csv << "frame_id,time_s,candidate_id,detect_valid,pnp_valid,"
           "tx_m,ty_m,tz_m,distance_m,rx_rad,ry_rad,rz_rad,reprojection_error_px"
        << endl;
    csv << fixed << setprecision(6);


    // 5.逐帧读取检测
    const int MAX_DEBUG_FRAMES = 0;      // 调试用：只跑前N帧，0=跑全部

    Mat frame, canva;
    int       frame_id          = 0;
    int       frames_with_armor = 0;
    int       total_armors      = 0;
    PoseStats stats;

    while (cap.read(frame)) {
        if (MAX_DEBUG_FRAMES > 0 && frame_id >= MAX_DEBUG_FRAMES) {
            break;
        }

        if (frame.empty()) {
            cerr << "读到空帧，frame_id = " << frame_id << endl;
            break;
        }

        if (frame.cols!=1280||frame.rows!=1024) {
            cerr << "帧 " << frame_id << " 的尺寸为 "<< frame.cols << " × " << frame.rows
                 << "，与标定尺寸 1280 × 1024 不符，停止位姿解算。" << endl;
            return 1;
        }

        //处理这一帧（检测、画框、位姿统计、写CSV、重投影误差、粗略深度核对、敏感性实验）
        int n = process_frame(frame, frame_id, video, csv, stats, canva);

        if (n > 0) {
            frames_with_armor++;
            total_armors += n;
        }

        //每333帧存一张结果图，以便检查
        if (frame_id % 333 == 0) {
            string name = "frame_" + to_string(frame_id) + ".png";
            imwrite("output/" + name, canva);
            cout << endl;
            cout << "存图 " << name << "（候选 " << n << " 个）" << endl;
            cout << endl;
        }

        //把该帧写出到结果视频上
        video_writer.write(canva);

        frame_id++;
    }


    // 6.判断视频是否提前结束
    const bool reached_debug_limit = MAX_DEBUG_FRAMES > 0 && frame_id >= MAX_DEBUG_FRAMES;
    if (!reached_debug_limit && video.frame_count>0 && frame_id<video.frame_count) {
        cerr << "视频疑似提前结束，报告帧数 " << video.frame_count
             << "，实际处理帧数 " << frame_id << "。" << endl;

        if (frame_id > 0) {
            cerr << "最后成功处理的帧号为 " << frame_id - 1 << "。" << endl;
        }
        else {
            cerr << "没有成功处理任何帧。" << endl;
        }
    }


    // 7.收尾
    video_writer.release();
    csv.close();
    cout << "逐帧数据：output/pose_log.csv" << endl;

    
    // 8.打印汇总
    cout << "=========================" << endl;
    cout << "总帧数     " << frame_id << endl;
    cout << "有检出的帧  " << frames_with_armor << endl;
    cout << "装甲板总数  " << total_armors << endl;

    if (frame_id > 0) {
        cout << "检出率     "
             << (100.0 * frames_with_armor / frame_id) << " %" << endl;
    }

    cout << endl;
    cout << "位姿解算成功: " << stats.ok   << " 次" << endl;
    cout << "位姿解算失败: " << stats.fail << " 次" << endl;

    if (stats.tz_count > 0) {
        cout << "tz 范围: " << stats.tz_min << " ~ " << stats.tz_max
             << " 米，平均 " << stats.tz_sum / stats.tz_count << " 米" << endl;
    }

    return 0;
}