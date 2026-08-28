#ifndef DRT_EPIPOLAR_FACTOR_H
#define DRT_EPIPOLAR_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include "utils/sophusExtUtils.hpp"

namespace DRT {

// Structureless epipolar (coplanarity) factor — paper Eq. 13-19.
// Scalar residual: r = A^T [C]_x B
//   A = R_j * Rbc * z_j        (bearing j, global frame)
//   B = R_i * Rbc * z_i        (bearing i, global frame)
//   t = (p_i + R_i*pbc) - (p_j + R_j*pbc)     (= p_Ci - p_Cj)
//   C = t / ||t||
// Connects the two pose blocks [omega, p] (size 6 each) only.
// Jacobians are tangent-space (right-perturbation rotation, body-frame position:
// position part chained through R) to match PoseLocalParameterization + ImuIntegFactor.
class EpipolarFactor : public ceres::SizedCostFunction<1, 6, 6> {
public:
    EpipolarFactor(const Eigen::Vector3d& z_i, const Eigen::Vector3d& z_j,
                   const Eigen::Matrix3d& Rbc, const Eigen::Vector3d& pbc,
                   double weight = 1.0)
        : z_i_(z_i), z_j_(z_j), Rbc_(Rbc), pbc_(pbc), weight_(weight) {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals, double** jacobians) const {
        Eigen::Map<const Eigen::Vector3d> omega_i(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> p_i(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> omega_j(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> p_j(parameters[1] + 3);

        Eigen::Matrix3d R_i = Sophus::SO3d::exp(omega_i).matrix();
        Eigen::Matrix3d R_j = Sophus::SO3d::exp(omega_j).matrix();

        Eigen::Vector3d A = R_j * Rbc_ * z_j_;
        Eigen::Vector3d B = R_i * Rbc_ * z_i_;
        Eigen::Vector3d t = (p_i + R_i * pbc_) - (p_j + R_j * pbc_);
        double t_norm = t.norm();
        if (t_norm < 1e-4) {                 // additiono degenerate baseline mid-optimization
            residuals[0] = 0.0;
            if (jacobians) {
                if (jacobians[0]) Eigen::Map<Eigen::Matrix<double,1,6,Eigen::RowMajor>>(jacobians[0]).setZero();
                if (jacobians[1]) Eigen::Map<Eigen::Matrix<double,1,6,Eigen::RowMajor>>(jacobians[1]).setZero();
            }
            return true;
        }
        Eigen::Vector3d C = t / t_norm;

        residuals[0] = weight_ * A.dot(C.cross(B));       // A^T [C]_x B

        if (!jacobians) return true;

        Eigen::Matrix3d Cx = Sophus::SO3d::hat(C);
        Eigen::Matrix3d Bx = Sophus::SO3d::hat(B);
        Eigen::RowVector3d dr_dA = (Cx * B).transpose();          // 1x3
        Eigen::RowVector3d dr_dB = A.transpose() * Cx;            // 1x3
        Eigen::RowVector3d dr_dC = -A.transpose() * Bx;           // 1x3
        Eigen::Matrix3d dC_dt =
            (Eigen::Matrix3d::Identity() - C * C.transpose()) / t_norm;   // 3x3
        Eigen::RowVector3d dr_dt = dr_dC * dC_dt;                 // 1x3

        if (jacobians[0]) {                                       // pose_i
            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> J(jacobians[0]);
            J.block<1,3>(0,0) = dr_dB * (-R_i * Sophus::SO3d::hat(Rbc_ * z_i_))
                              + dr_dt * (-R_i * Sophus::SO3d::hat(pbc_));   // d/dphi_i
            J.block<1,3>(0,3) = dr_dt * R_i;                                // d/dp_i (body)
            J *= weight_;
        }
        if (jacobians[1]) {                                       // pose_j
            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> J(jacobians[1]);
            J.block<1,3>(0,0) = dr_dA * (-R_j * Sophus::SO3d::hat(Rbc_ * z_j_))
                              + dr_dt * ( R_j * Sophus::SO3d::hat(pbc_));   // d/dphi_j
            J.block<1,3>(0,3) = -dr_dt * R_j;                               // d/dp_j (body)
            J *= weight_;
        }
        return true;
    }

private:
    Eigen::Vector3d z_i_, z_j_, pbc_;
    Eigen::Matrix3d Rbc_;
    double weight_;
};

}  // namespace DRT
#endif  // DRT_EPIPOLAR_FACTOR_H