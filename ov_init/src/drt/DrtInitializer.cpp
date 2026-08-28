#include "DrtInitializer.h"

#include "core/drt_glue.h"
#include "utils/helper.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "types/IMU.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

#include <algorithm>
#include <set>

using namespace ov_core;
using namespace ov_type;
using namespace ov_init;

Eigen::Vector3d G;   // definition for DRT's extern G

DrtInitializer::DrtInitializer(const InertialInitializerOptions &params_, std::shared_ptr<ov_core::FeatureDatabase> db,
                               std::shared_ptr<std::vector<ov_core::ImuData>> imu_data_)
    : params(params_), _db(db), imu_data(imu_data_) {
  G = Eigen::Vector3d(0.0, 0.0, params_.gravity_mag);
}

bool DrtInitializer::initialize(double &timestamp, Eigen::MatrixXd &covariance,
                                std::vector<std::shared_ptr<ov_type::Type>> &order, std::shared_ptr<ov_type::IMU> &_imu,
                                std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
                                std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &_features_SLAM) {

  static const int MIN_FRAMES = 10; // DRT's own harness requires 10

  // ---- window bounds
  double newest_cam_time = -1;
  for (auto const &feat : _db->get_internal_data())
    for (auto const &camtimepair : feat.second->timestamps)
      for (auto const &time : camtimepair.second)
        newest_cam_time = std::max(newest_cam_time, time);
  double oldest_time = newest_cam_time - params.init_window_time;
  if (newest_cam_time < 0 || oldest_time < 0)
    return false;
  _db->cleanup_measurements(oldest_time);
  if (imu_data->size() < 2)
    return false;

  // ---- copy features (the db keeps growing on the tracking thread)
  std::unordered_map<size_t, std::shared_ptr<Feature>> features;
  for (const auto &feat : _db->get_internal_data()) {
    auto f = std::make_shared<Feature>();
    f->featid = feat.second->featid;
    f->uvs_norm = feat.second->uvs_norm;
    f->timestamps = feat.second->timestamps;
    features.insert({feat.first, f});
  }

  // ---- every cam0 frame in the window; DRT picks its own keyframes
  std::set<double> all_times;
  for (auto const &f : features) {
    if (f.second->timestamps.find(0) == f.second->timestamps.end())
      continue;
    for (double t : f.second->timestamps.at(0))
      if (t >= oldest_time)
        all_times.insert(t);
  }
  std::vector<double> stamps(all_times.begin(), all_times.end());
  int N = (int)stamps.size();
  if (N < MIN_FRAMES) {
    PRINT_DEBUG(YELLOW "[init-drt]: only %d frames in window\n" RESET, N);
    return false;
  }

  // ---- features per frame (DRT reads only components 0,1,2)
  std::vector<std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>> feats(N);
  for (int i = 0; i < N; i++) {
    for (auto const &f : features) {
      auto it = f.second->timestamps.find(0);
      if (it == f.second->timestamps.end())
        continue;
      auto pos = std::find(it->second.begin(), it->second.end(), stamps[i]);
      if (pos == it->second.end())
        continue;
      size_t idx = std::distance(it->second.begin(), pos);
      Eigen::Matrix<double, 7, 1> p = Eigen::Matrix<double, 7, 1>::Zero();
      p(0) = f.second->uvs_norm.at(0).at(idx)(0);
      p(1) = f.second->uvs_norm.at(0).at(idx)(1);
      p(2) = 1.0;
      feats[i][(int)f.second->featid].emplace_back(0, p);
    }
  }

  // ---- flat IMU stream, shifted into camera clock
  std::vector<ImuData> readings =
      InitializerHelper::select_imu_readings(*imu_data, stamps.front() + params.calib_camimu_dt,
                                             stamps.back() + params.calib_camimu_dt);
  if (readings.size() < 2) {
    PRINT_DEBUG(YELLOW "[init-drt]: only %zu imu readings\n" RESET, readings.size());
    return false;
  }
  std::vector<DrtImuStamped> imu_stream;
  for (auto const &m : readings)
    imu_stream.push_back({m.wm, m.am, m.timestamp - params.calib_camimu_dt});
  // ---- reject windows with too little rotation: without it, accel bias and
  //      gravity are not separable (same check OpenVINS's dynamic init does)
  double theta_deg = 0.0;
  for (size_t k = 0; k + 1 < imu_stream.size(); k++) {
    double dt = imu_stream[k + 1].t - imu_stream[k].t;
    if (dt > 0)
      theta_deg += (0.5 * (imu_stream[k].gyr + imu_stream[k + 1].gyr) * dt).norm();
  }
  theta_deg *= 180.0 / M_PI;
  if (theta_deg < params.init_dyn_min_deg) {
    PRINT_DEBUG(YELLOW "[init-drt]: only %.2f deg rotation (%.2f thresh)\n" RESET, theta_deg, params.init_dyn_min_deg);
    return false;
  }
  PRINT_DEBUG("[init-drt]: window rotation %.2f deg\n", theta_deg);
  // ---- extrinsics: OpenVINS stores IMU->cam, DRT wants cam->IMU
  Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(0).block(0, 0, 4, 1);
  Eigen::Vector3d p_IinC = params.camera_extrinsics.at(0).block(4, 0, 3, 1);
  Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);
  Eigen::Matrix3d Ric = R_ItoC.transpose();
  Eigen::Vector3d Tic = -R_ItoC.transpose() * p_IinC;

  // ---- run DRT + VI-BA
  // DRT was tuned against its own noise values (drt-vio-init/config/euroc.yaml).
  // OpenVINS uses EuRoC's datasheet accel noise density, which is 10x larger and
  // starves the accelerometer constraints that determine gravity.
  const double drt_gyr_n = 1.7e-4, drt_acc_n = 2.0e-4;
  const double drt_gyr_w = 1.9393e-5, drt_acc_w = 5.0e-3;
  DrtInitOut r = RunDrtInit(Ric, Tic, drt_gyr_n, drt_acc_n, drt_gyr_w, drt_acc_w,
                            params.init_drt_vis_weight, stamps, feats, imu_stream, MIN_FRAMES);
  if (!r.ok) {
    PRINT_DEBUG(YELLOW "[init-drt]: DRT failed (fed %d frames)\n" RESET, N);
    return false;
  }

  // ---- hand back the newest state (Hamilton R_wb -> JPL q_GtoI)
  size_t last = r.R.size() - 1;
  if (_imu == nullptr)
    _imu = std::make_shared<IMU>();
  Eigen::VectorXd imu_state = Eigen::VectorXd::Zero(16);
  imu_state.block(0, 0, 4, 1) = rot_2_quat(r.R[last].transpose());
  imu_state.block(4, 0, 3, 1) = Eigen::Vector3d::Zero(); // anchor position is zeroed
  imu_state.block(7, 0, 3, 1) = r.V[last];
  imu_state.block(10, 0, 3, 1) = r.bg;
  imu_state.block(13, 0, 3, 1) = r.ba;
  _imu->set_value(imu_state);
  _imu->set_fej(imu_state);
  timestamp = r.stamps[last];

  // ---- M4 placeholder: fixed prior, same shape as StaticInitializer's
  order.clear();
  order.push_back(_imu);
  if (r.cov.rows() == 15) {
    covariance = r.cov;
  } else {
    PRINT_DEBUG(YELLOW "[init-drt]: covariance recovery failed, using fallback prior\n" RESET);
    covariance = Eigen::MatrixXd::Zero(_imu->size(), _imu->size());
    covariance.block(0, 0, 3, 3) = std::pow(0.02, 2) * Eigen::Matrix3d::Identity();
    covariance.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();
    covariance.block(6, 6, 3, 3) = std::pow(0.10, 2) * Eigen::Matrix3d::Identity();
    covariance.block(9, 9, 3, 3) = std::pow(0.02, 2) * Eigen::Matrix3d::Identity();
    covariance.block(12, 12, 3, 3) = std::pow(0.10, 2) * Eigen::Matrix3d::Identity();
  }
  // Inflate while preserving correlation structure: C' = S C S
  Eigen::VectorXd s = Eigen::VectorXd::Ones(15);
  s.segment(0, 3).setConstant(std::sqrt(params.init_dyn_inflation_orientation));
  s.segment(6, 3).setConstant(std::sqrt(params.init_dyn_inflation_velocity));
  s.segment(9, 3).setConstant(std::sqrt(params.init_dyn_inflation_bias_gyro));
  s.segment(12, 3).setConstant(std::sqrt(params.init_dyn_inflation_bias_accel));
  covariance = s.asDiagonal() * covariance * s.asDiagonal();
  PRINT_INFO(GREEN "[init-drt]: success, %zu of %d frames, |v| = %.3f\n" RESET, r.R.size(), N, r.V[last].norm());
  return true;
}