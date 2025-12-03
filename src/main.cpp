#include <string>
#include <iostream>
#include <Eigen/Dense>
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

    return 0;
}