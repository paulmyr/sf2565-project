#include <string>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <algorithm>
#include <CLI11.hpp>
#include "arnoldi.h"
#include "lanczos.h"
#include <fstream>
#include <stdexcept>

using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

/* expecting matrix file of the sort
 * rows cols nnz
 * r1 c1 v1
 * r2 c2 v2
 * ...
 */
SpMat read_sparse_input_file(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("could not open file " + filename);
    }

    int rows, cols, nnz;
    if (!(in >> rows >> cols >> nnz)) {
        throw std::runtime_error("could not read matrix dimensions/nnz from " + filename);
    }

    std::vector<Triplet> coefficients;
    coefficients.reserve(nnz);

    for (int k = 0; k < nnz; ++k) {
        int r, c;
        double v;
        if (!(in >> r >> c >> v)) {
            throw std::runtime_error("failed to read triplet line " + std::to_string(k+1));
        }
        coefficients.push_back(Triplet(r, c, v));
    }

    SpMat A(rows, cols);
    A.setFromTriplets(coefficients.begin(), coefficients.end());

    return A;
}

int main(int argc, char **argv) {
    CLI::App app{"Arnoldi Iteration"};

    std::string input_file;
    app.add_option("--input-file,-i", input_file,
        "Input file path. Expects file in TODO SPECIFY FORMAT");

    CLI11_PARSE(app, argc, argv);

    SpMat A;

    try {
        A = read_sparse_input_file(input_file);
        std::cout << "loaded sparse matrix: " << A.rows() << "x" << A.cols()
                  << " with " << A.nonZeros() << " non-zeros.\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    auto test = arnoldi::solve(A, 3);
    std::cout << test.krylov_basis << std::endl;
    std::cout << test.hessenberg << std::endl;
    std::cout << test.iterations << std::endl;


    // Lanczos
    SpMat U = A;

    Eigen::VectorXd k0 = Eigen::VectorXd::Ones(10000);
    Eigen::VectorXd q0 = Eigen::VectorXd::Ones(10000);

    int m = 2;
    auto result = lanczos::solve(U, k0, q0, m);

    // Results from lanczos
    std::cout << "Tridiagonal T:\n" << result.T << "\n\n";
    std::cout << "Eigenvalue approximations:\n"
              << result.eigenvalues.transpose() << "\n\n";
    std::cout << "Iterations: " << result.iterations << "\n";

    return 0;
}