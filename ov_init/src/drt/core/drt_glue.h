#pragma once
#include <vector>
#include <map>
#include <utility>
#include <Eigen/Dense>

// One raw IMU sample, timestamped in CAMERA clock.
struct DrtImuStamped { Eigen::Vector3d gyr, acc; double t; };

// Plain-typed bridge to DRT+VI-BA (keeps DRT headers out of DrtInitializer.cpp).
struct DrtInitOut {
    bool ok = false;
    std::vector<double> stamps;              // frames DRT actually selected
    std::vector<Eigen::Matrix3d> R;          // body-in-world, gravity-aligned
    std::vector<Eigen::Vector3d> P, V;
    Eigen::Vector3d bg, ba, g;
    Eigen::MatrixXd cov;  
};

DrtInitOut RunDrtInit(
    const Eigen::Matrix3d& Ric, const Eigen::Vector3d& Tic,
    double gyr_n, double acc_n, double gyr_w, double acc_w, double vis_weight,
    const std::vector<double>& stamps,
    const std::vector<std::map<int, std::vector<std::pair<int, Eigen::Matrix<double,7,1>>>>>& feats,
    const std::vector<DrtImuStamped>& imu_stream,
    int min_frames);