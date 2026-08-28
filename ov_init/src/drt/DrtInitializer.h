#ifndef OV_INIT_DRTINITIALIZER_H
#define OV_INIT_DRTINITIALIZER_H

#include "init/InertialInitializerOptions.h"

namespace ov_core {
class FeatureDatabase;
struct ImuData;
} // namespace ov_core
namespace ov_type {
class Type;
class IMU;
class PoseJPL;
class Landmark;
} // namespace ov_type

namespace ov_init {

//class DynamicInitializer;

/**
 * @brief DRT + structureless VI-BA initializer.
 */
class DrtInitializer {
public:
  DrtInitializer(const InertialInitializerOptions &params_, std::shared_ptr<ov_core::FeatureDatabase> db,
                 std::shared_ptr<std::vector<ov_core::ImuData>> imu_data_);

  bool initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                  std::shared_ptr<ov_type::IMU> &_imu, std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
                  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &_features_SLAM);

private:
  InertialInitializerOptions params;
  std::shared_ptr<ov_core::FeatureDatabase> _db;
  std::shared_ptr<std::vector<ov_core::ImuData>> imu_data;

  /// TEMPORARY: delegate target for M1 plumbing test
  //std::shared_ptr<DynamicInitializer> _fallback;
};

} // namespace ov_init

#endif // OV_INIT_DRTINITIALIZER_H