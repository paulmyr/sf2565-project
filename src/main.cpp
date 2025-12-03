#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <algorithm>
#include <CLI11.hpp>
#include "arnoldi.h"
#include "lanczos.h"


void read_input_file(std::string filename) {
    // TODO: everything

    return;
}

int main(int argc, char **argv) {
    CLI::App app{"Arnoldi Iteration"};

    std::string input_file;
    app.add_option("--input-file,-i", input_file,
        "Input file path. Expects file in TODO SPECIFY FORMAT");

    CLI11_PARSE(app, argc, argv);

    Eigen::MatrixXd A(4, 4);

    if (input_file.empty()) {
        A << 1, 2, 0, 0,
             2, 3, 1, 0,
             0, 1, 2, 1,
             0, 0, 1, 4;
        std::cout << "Example Eigen matrix:\n" << A << std::endl;
    }

    auto test = arnoldi::solve(A, 3);
    std::cout << test.krylov_basis << std::endl;
    std::cout << test.hessenberg << std::endl;
    std::cout << test.iterations << std::endl;


    // Lanczos
    Eigen::MatrixXd U(4, 4);
    U << 1, 2, 0, 0,
         2, 3, 1, 0,
         0, 1, 2, 1,
         0, 0, 1, 4;

    Eigen::VectorXd k0 = Eigen::VectorXd::Ones(4);
    Eigen::VectorXd q0 = Eigen::VectorXd::Ones(4);

    int m = 2;
    auto result = lanczos::solve(U, k0, q0, m);

    // Results from lanczos
    std::cout << "Tridiagonal T:\n" << result.T << "\n\n";
    std::cout << "Eigenvalue approximations:\n"
              << result.eigenvalues.transpose() << "\n\n";
    std::cout << "Iterations: " << result.iterations << "\n";

    // True eigs
    Eigen::EigenSolver<Eigen::MatrixXd> full(U);
    std::cout << "\nTrue eigenvalues of U:\n"
              << full.eigenvalues() << "\n";


    // Simon Test for new Lanczos
    std::cout << "Simon Test for new Lanczos" "\n";

    const int n = 4;
    Eigen::MatrixXd V(n, n);
    V << 1, 2, 0, 0,
         2, 3, 1, 0,
         0, 1, 2, 1,
         0, 0, 1, 4;

    std::cout << "V =\n" << V << "\n\n";

    Eigen::VectorXd k1 = Eigen::VectorXd::Random(n);
    Eigen::VectorXd q1 = Eigen::VectorXd::Random(n);

    auto res = lanczos::solve(U, k1, q1, n);

    std::cout << "T (tridiagonal reduction) =\n" << res.T << "\n\n";

    // Eigenvalues of V
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