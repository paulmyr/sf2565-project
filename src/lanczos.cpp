#include "lanczos.h"
#include <cassert>

namespace lanczos {
    LanczosResult solve(
        const Eigen::MatrixXd& U,
        const Eigen::VectorXd& k0,
        const Eigen::VectorXd& q0,
        int m
    ) {
        std::vector<Eigen::VectorXd> k(m + 1);
        std::vector<Eigen::VectorXd> q(m + 1);

        Eigen::VectorXd alpha = Eigen::VectorXd::Zero(m);
        Eigen::VectorXd beta  = Eigen::VectorXd::Zero(m-1);

        k[0] = k0;
        q[0] = q0;

        // First two stages
        alpha(0) = (q[0].transpose() * U * k[0] / q[0].dot(k[0])).value();
        k[1] = U * k[0] - alpha[0] * k[0];
        q[1] = U * q[0] - alpha[0] * q[0];

        alpha(1) = (q[1].transpose() * U * k[1] / q[1].dot(k[1])).value();
        beta(0) = (q[0].transpose() * U * k[1] / q[0].dot(k[0])).value();

        // Iteration for the rest
        for (int i = 2; i < m; ++i) {
            k[i] = U * k[i-1] - alpha[i-1] * k[i-1] - beta[i-2] * k[i-2];
            q[i] = U * q[i-1] - alpha[i-1] * q[i-1] - beta[i-2] * q[i-2];

            alpha(i) = (q[i].transpose() * U * k[i] / q[i].dot(k[i])).value();
            beta(i-1) = (q[i-1].transpose() * U * k[i] / q[i-1].dot(k[i-1])).value();
        }

        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);

        for (int i = 0; i < m; ++i) {
            T(i, i) = alpha(i);
            if (i > 0) T(i, i - 1) = 1.0;
            if (i < m - 1) T(i, i + 1) = beta(i);
        }

        Eigen::EigenSolver<Eigen::MatrixXd> solver(T);
        Eigen::VectorXd eigs = solver.eigenvalues().real();

        return LanczosResult{
            T,
            eigs,
            m
        };
    }

}