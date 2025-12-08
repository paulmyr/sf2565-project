#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <algorithm>
#include <CLI11.hpp>
#include "lanczos.h"
#include <fstream>
#include <stdexcept>

Eigen::MatrixXd read_input_file(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("could not open file:" + filename);
    }
    /* expecting matrix file of the sort
     * n (integer)
     * a_11 a_12 ... a_1n (doubles)
     * ...
     * a_n1 a_n2 ... a_nn
     */
    int n;
    in >> n;
    if (!in || n <= 0) {
        throw std::runtime_error("invalid matrix size in file: " + filename);
    }
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val;
            in >> val;
            if (!in) {
                throw std::runtime_error("failed to read entry (" +
                                         std::to_string(i) + "," +
                                         std::to_string(j) + ") from file: " + filename);
            }
            A(i, j) = val;
        }
    }
    return A;
}

int main(int argc, char **argv) {
    // Test for Lanczos
    std::cout << "Simon Test for new Lanczos" "\n";

    const int n = 4;
    Eigen::MatrixXd U(n, n);
    U << 1, 2, 0, 0,
         2, 3, 1, 0,
         0, 1, 2, 1,
         0, 0, 1, 4;

    std::cout << "U =\n" << U << "\n\n";

    Eigen::VectorXd k1 = Eigen::VectorXd::Random(n);
    Eigen::VectorXd q1 = Eigen::VectorXd::Random(n);

    const int m = 4;
    auto res = lanczos::solve(U, k1, q1, m);

    std::cout << "T (tridiagonal reduction) =\n" << res.T << "\n\n";

    // Eigenvalues of U
    Eigen::EigenSolver<Eigen::MatrixXd> esU(U);
    Eigen::VectorXd lambda_U = esU.eigenvalues().real();

    // Eigenvalues of T (already computed in res.eigenvalues)
    Eigen::VectorXd lambda_T = res.eigenvalues;

    std::vector<double> eigU(lambda_U.data(), lambda_U.data() + lambda_U.size());
    std::vector<double> eigT(lambda_T.data(), lambda_T.data() + lambda_T.size());
    std::sort(eigU.begin(), eigU.end());
    std::sort(eigT.begin(), eigT.end());

    std::cout << "Eigenvalues of U (sorted): ";
    for (double x : eigU) std::cout << x << " ";
    std::cout << "\nEigenvalues of T (sorted): ";
    for (double x : eigT) std::cout << x << " ";
    std::cout << "\n";


    std::cout << "Pointwise absolute errors:\n";
    for (int i = 0; i < eigU.size(); ++i) {
        double err = std::abs(eigT[i] - eigU[i]);
        std::cout << "i = " << i
                  << ": |λ_T - λ_U| = " << err << "\n";
    }

    return 0;
}