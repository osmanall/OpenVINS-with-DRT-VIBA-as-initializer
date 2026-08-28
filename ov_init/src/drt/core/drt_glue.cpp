#include "drt_glue.h"
#include "initMethod/drtLooselyCoupled.h"
#include "IMU/imuPreintegrated.hpp"
#include "IMU/basicTypes.hpp"
#include "utils/eigenUtils.hpp"
#include <cmath>

DrtInitOut RunDrtInit(
    const Eigen::Matrix3d& Ric, const Eigen::Vector3d& Tic,
    double gyr_n, double acc_n, double gyr_w, double acc_w, double vis_weight,
    const std::vector<double>& stamps,
    const std::vector<std::map<int, std::vector<std::pair<int, Eigen::Matrix<double,7,1>>>>>& feats,
    const std::vector<DrtImuStamped>& imu_stream,
    int min_frames)
{
    DrtInitOut out;
    if ((int)stamps.size() < 2 || imu_stream.size() < 2) return out;

    // Nominal IMU period, used to scale the continuous-time noise densities.
    // NOTE: calib must outlive every IMUPreintegrated that points at it.
    double dt0 = imu_stream[1].t - imu_stream[0].t;
    double sf = std::sqrt(1.0 / std::max(dt0, 1e-4));
    vio::IMUCalibParam calib(Ric, Tic, gyr_n * sf, acc_n * sf, gyr_w / sf, acc_w / sf);

    DRT::drtLooselyCoupled drt(Ric, Tic);
    drt.vis_weight_ = vis_weight;
    std::vector<double> used;
    double last_t = -1;

    for (int i = 0; i < (int)stamps.size(); i++) {
        FeatureTrackerResulst image;
        for (auto& f : feats[i])
            for (auto& obs : f.second)
                image[f.first].emplace_back(obs.first, obs.second);

        // DRT applies its own 0.22 s selection rule here
        if (!drt.addFeatureCheckParallax(stamps[i], image, 0.0))
            continue;

        if (last_t > 0) {
            vio::IMUBias bias;
            vio::IMUPreintegrated pre(bias, &calib, last_t, stamps[i]);
            // Midpoint integration with interpolation at both boundaries, so the
            // integrated span is exactly [last_t, stamps[i]] -- matches DRT's own
            // harness (app/main.cpp). Truncating to whole samples loses ~2.3% of
            // each interval, which biases dP_/dV_ and hence the gravity solve.
            for (size_t k = 0; k + 1 < imu_stream.size(); k++) {
                double t0 = imu_stream[k].t, t1 = imu_stream[k + 1].t;
                if (t1 <= last_t) continue;      // entirely before the interval
                if (t0 >= stamps[i]) break;      // entirely after it
                Eigen::Vector3d g0 = imu_stream[k].gyr, g1 = imu_stream[k + 1].gyr;
                Eigen::Vector3d a0 = imu_stream[k].acc, a1 = imu_stream[k + 1].acc;
                double tab = t1 - t0;
                if (tab <= 0) continue;
                double ta = std::max(t0, last_t);      // clipped start
                double tb = std::min(t1, stamps[i]);   // clipped end
                double dt = tb - ta;
                if (dt <= 0) continue;
                // linear interpolation to the clipped endpoints, then average
                double wa = (ta - t0) / tab, wb = (tb - t0) / tab;
                Eigen::Vector3d ga = g0 + (g1 - g0) * wa, gb = g0 + (g1 - g0) * wb;
                Eigen::Vector3d aa = a0 + (a1 - a0) * wa, ab = a0 + (a1 - a0) * wb;
                pre.integrate_new_measurement(0.5 * (ga + gb), 0.5 * (aa + ab), dt);
            }
            drt.addImuMeasure(pre);
        }
        last_t = stamps[i];
        used.push_back(stamps[i]);
    }

    if ((int)used.size() < min_frames) return out;
    if (!drt.checkAccError()) return out;
    if (!drt.process()) return out;
    if (drt.rotation.size() != used.size()) return out;

    // gravity-align, then cancel the first frame's yaw
    Eigen::Matrix3d R0 = Utility::g2R(drt.gravity);
    double yaw = Utility::R2ypr(R0 * drt.rotation[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    out.g = R0 * drt.gravity;
    for (size_t i = 0; i < used.size(); i++) {
        out.R.push_back(R0 * drt.rotation[i]);
        out.P.push_back(R0 * drt.position[i]);
        out.V.push_back(R0 * drt.velocity[i]);
    }
    out.stamps = used;
        // Rotate the covariance into the gravity-aligned frame we hand over.
    // dtheta is body-frame (unchanged), dp is body->global, dv is aligned->final.
    if (drt.cov_last_.rows() == 15) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Identity(15, 15);
        J.block<3, 3>(3, 3) = out.R.back();   // dp_body -> global
        J.block<3, 3>(6, 6) = R0;             // dv aligned -> final
        out.cov = J * drt.cov_last_ * J.transpose();
    }
    out.bg = drt.biasg;
    out.ba = drt.biasa;
    out.ok = true;
    return out;
}