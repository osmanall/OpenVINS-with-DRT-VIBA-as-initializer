#ifndef DRT_POSE_LOCAL_PARAM_H
#define DRT_POSE_LOCAL_PARAM_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include "utils/sophusExtUtils.hpp"

namespace DRT {

// Pose block layout: [ omega(3) = log(R), p(3) ]
//  - Plus: RIGHT rotation perturbation R <- R*exp(dphi), BODY-frame translation p <- p + R*dp
//    (matches vio::ImuIntegFactor's Jacobians).
//  - ComputeJacobian = identity so the factor's tangent Jacobians pass through unchanged.
class PoseLocalParameterization : public ceres::LocalParameterization {
public:
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const {
        Eigen::Map<const Eigen::Vector3d> omega(x);
        Eigen::Map<const Eigen::Vector3d> p(x + 3);
        Eigen::Map<const Eigen::Vector3d> dphi(delta);
        Eigen::Map<const Eigen::Vector3d> dp(delta + 3);

        Sophus::SO3d R     = Sophus::SO3d::exp(omega);
        Sophus::SO3d R_new = R * Sophus::SO3d::exp(dphi);   // right perturbation

        Eigen::Map<Eigen::Vector3d> omega_new(x_plus_delta);
        Eigen::Map<Eigen::Vector3d> p_new(x_plus_delta + 3);
        omega_new = R_new.log();
        p_new     = p + R * dp;                             // body-frame translation
        return true;
    }

    virtual bool ComputeJacobian(const double* /*x*/, double* jacobian) const {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize()  const { return 6; }
};

}  // namespace DRT

#endif  // DRT_POSE_LOCAL_PARAM_H