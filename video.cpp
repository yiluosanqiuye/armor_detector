#include "video.h"
#include "armor.h"
#include "pose.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
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
    Armor armor;
    Pose  pose;
};


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
    cout << "视频路径：" << path              << endl;
    cout << "宽 = "     << video.width       << endl;
    cout << "高 = "     << video.height      << endl;
    cout << "帧率 = "   << video.fps         << endl;
    cout << "总帧数 = " << video.frame_count << endl;
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
    for (int i = 0; i < (int)candidates.size(); i++) {
        const Armor &armor = candidates[i].armor;
        const Pose  &pose  = candidates[i].pose;

        string text;
        Scalar color;

        //检测失败（四角点不可用）
        if (armor.valid == false) {
            text  = "#" + to_string(i) + "   DET FAIL";
            color = Scalar(0, 0, 255);
        }
        //检测和位姿解算都成功
        else if (pose.valid == true) {
            ostringstream oss;
            oss << "#" << i << "   PnP OK   distance = "
                << fixed << setprecision(2) << pose.distance << " m";
            text  = oss.str();
            color = Scalar(0, 255, 0);
        }
        //检测成功但位姿解算失败
        else {
            text  = "#" + to_string(i) + "   PnP FAIL";
            color = Scalar(0, 0, 255);
        }

        putText(canva, text, Point(x, y),
                FONT_HERSHEY_SIMPLEX, 0.55, color, 1, LINE_AA);

        y += line_h;
    }
}


//处理单帧图像，并返回这一帧的候选装甲板数量
static int process_frame(Mat &frame, int frame_id, const Video &video,ofstream &csv, PoseStats &stats, Mat &canva) {
//每帧

    //检测
    vector<Armor> armors = detect_armors(frame);

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

        //检测是否有效（装甲板四角点是否可用）
        int detect_ok = armor.valid ? 1 : 0;

        //只有检测有效才解算位姿
        Pose pose;
        if (detect_ok == 1) {
            pose = estimate_pose(armor.corners);
        }

        //打包成候选（无论有效与否都存）
        int candidate_id = (int)candidates.size();
        candidates.push_back({armor, pose});

        //画框+标号
        photo.draw_armor(canva, armor);
        draw_candidate_id(canva, armor, candidate_id);

        //位姿统计
        stats.add(pose);

        //写CSV
        csv << frame_id << "," << (frame_id / video.fps) << ","
            << candidate_id << "," << detect_ok << ",";

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

        candidate_id++;
    }

    // 画信息面板
    draw_information_panel(canva, frame_id, video, candidates);

    return (int)candidates.size();
}


// ---------------------------------------视频主流程------------------------------------------------
int run_video(const string &path) {

    //1.打开视频
    VideoCapture cap;
    if (!open_video(cap, path)) {
        return 1;
    }


    //2.读取并打印属性
    Video video = read_video_information(cap);
    print_video_information(path, video);


    //3.创建输出视频
    string out_path = "output/result.avi";
    VideoWriter video_writer(out_path,
                             VideoWriter::fourcc('M', 'J', 'P', 'G'),
                             video.fps,
                             Size(video.width, video.height));

    if (!video_writer.isOpened()) {
        cerr << "输出视频创建失败：" << out_path << endl;
        return 1;
    }
    cout << "输出视频：" << out_path << endl;
    cout << endl;


    //4.打开逐帧数据文件csv
    ofstream csv("output/pose_log.csv");
    if (!csv.is_open()) {
        cerr << "无法创建 output/pose_log.csv" << endl;
        return 1;
    }

    csv << "frame_id,time_s,candidate_id,detect_valid,pnp_valid,"
           "tx_m,ty_m,tz_m,distance_m,rx_rad,ry_rad,rz_rad,reprojection_error_px"
        << endl;
    csv << fixed << setprecision(6);


    //5.逐帧读取检测
    const int MAX_FRAMES = 0;      // 调试用：只跑前N帧，0=跑全部

    Mat frame, canva;
    int       frame_id          = 0;
    int       frames_with_armor = 0;
    int       total_armors      = 0;
    PoseStats stats;

    while (cap.read(frame)) {
        if (MAX_FRAMES > 0 && frame_id >= MAX_FRAMES) {
            break;
        }

        if (frame.empty()) {
            cerr << "读到空帧，frame_id = " << frame_id << endl;
            break;
        }

        //处理这一帧（检测、画框、位姿统计、写CSV）
        int n = process_frame(frame, frame_id, video, csv, stats, canva);

        if (n > 0) {
            frames_with_armor++;
            total_armors += n;
        }

        //每333帧存一张结果图，以便检查
        if (frame_id % 333 == 0) {
            string name = "frame_" + to_string(frame_id) + ".png";
            imwrite("output/" + name, canva);
            cout << "存图 " << name << "（候选 " << n << " 个）" << endl;
        }

        video_writer.write(canva);
        frame_id++;
    }


    //6.收尾
    video_writer.release();
    csv.close();
    cout << "逐帧数据：output/pose_log.csv" << endl;

    
    //7.打印汇总
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