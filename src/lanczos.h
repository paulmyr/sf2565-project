#ifndef LANCZOS_H
#define LANCZOS_H

#include <Eigen/Dense>

namespace lanczos {
    struct LanczosResult {
        Eigen::MatrixXd T;
        Eigen::VectorXd eigenvalues;
        int iterations;
    };

    LanczosResult solve(
        const Eigen::MatrixXd& U,
        const Eigen::VectorXd& k1,
        const Eigen::VectorXd& q1,
        int m
    );
}

#endif