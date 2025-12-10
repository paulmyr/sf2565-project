#ifndef LANCZOS_H
#define LANCZOS_H

#include <vector>
#include <cassert>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

struct LanczosResult {
    Eigen::MatrixXd T;
    Eigen::VectorXd eigenvalues;
    int iterations;
};

namespace lanczos {
    template <typename TMatrix>
    LanczosResult solve(
        const TMatrix& U,
        const Eigen::VectorXd& k1,  // k_1
        const Eigen::VectorXd& q1,  // q_1  (stored as column; represents row q_1^T)
        const int m
    ) {

        assert(U.rows() == U.cols());
        assert(U.rows() == k1.size());
        assert(U.rows() == q1.size());
        assert(m >= 1);

        // We store k_1,...,k_m in k[0,...,m-1], and similarly for q.
        std::vector<Eigen::VectorXd> k(m);
        std::vector<Eigen::VectorXd> q(m);

        k[0] = k1;
        q[0] = q1;

        // α_1,...,α_m  → alpha(0,...,m-1)
        Eigen::VectorXd alpha = Eigen::VectorXd::Zero(m);

        // β_1,...,β_{m-1} → beta(0..m-2)
        // β_0 is never stored, it is 0 in the recurrence.
        Eigen::VectorXd beta  = Eigen::VectorXd::Zero(std::max(0, m - 1));

        // Computing α_1
        const double num = q[0].dot(U * k[0]);    // q_1*U*k_1
        const double denom = q[0].dot(k[0]);           // q_1*k_1
        assert(std::abs(denom) > 0.0000001);     // sanity check for debugging
        alpha(0) = num / denom;             // α_1

        // β_0 ← 0 (Don't need to store)

        // Iteration
        for (int i = 2; i <= m; ++i) {
            int j = i-1;
            k[j] = U * k[j-1] - alpha(j-1) * k[j-1];

            // q_{n−1} U is a row times matrix. Since we store q as columns,
            // we use U^T * q_{n−1} which corresponds to (q_{n−1} U)^T.
            q[j] = U.transpose() * q[j-1] - alpha(j-1) * q[j-1];

            // β_{n−2} term is only present for i ≥ 3
            if (i >= 3) {
                k[j] -= beta(j-2) * k[j-2];
                q[j] -= beta(j-2) * q[j-2];
            }

            double alpha_num = q[j].dot(U * k[j]);
            double alpha_denom = q[j].dot(k[j]);
            alpha(j) = alpha_num / alpha_denom;

            int beta_index = (i - 1) - 1;
            if (beta_index >= 0) {
                double beta_num = q[j-1].dot(U * k[j]);
                double beta_denom = q[j-1].dot(k[j-1]);
                beta(beta_index) = beta_num / beta_denom;
            }
        }

        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);

        for (int i = 0; i < m; ++i) {
            T(i, i) = alpha(i);
            if (i > 0) T(i, i - 1) = 1.0;
            if (i < m - 1) T(i, i + 1) = beta(i);
        }

        const Eigen::EigenSolver<Eigen::MatrixXd> solver(T);
        const Eigen::VectorXd eigs = solver.eigenvalues().real();

        return LanczosResult{
            T,
            eigs,
            m
        };
    }
}

#endif