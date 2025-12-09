#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include "lanczos.h"

// ---------------------- Types ----------------------

using SpMat   = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

// ---------------------- MANUAL SELECTION ----------------------

enum class MatrixType {
    DenseRandom,
    DenseFromFile,
    SparseFromFile
};
// -----------------------------------------------

// File paths (used only when needed)
const std::string DENSE_FILE  = "dense.txt";
const std::string SPARSE_FILE = "sparse.txt";

// Matrix parameters


// ---------------------- File Readers ----------------------

SpMat read_sparse_input_file(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Could not open file: " + filename);

    int rows, cols, nnz;
    in >> rows >> cols >> nnz;
    if (!in) throw std::runtime_error("Invalid sparse header");

    std::vector<Triplet> triplets;
    triplets.reserve(nnz);

    for (int k = 0; k < nnz; ++k) {
        int r, c;
        double v;
        in >> r >> c >> v;
        if (!in) throw std::runtime_error("Invalid sparse entry");
        triplets.emplace_back(r, c, v);
    }

    SpMat A(rows, cols);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

Eigen::MatrixXd read_dense_input_file(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Could not open file: " + filename);

    int n;
    in >> n;
    if (!in || n <= 0) throw std::runtime_error("Invalid matrix size");

    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            in >> A(i, j);

    return A;
}

// ---------------------- Eigenvalue Comparison ----------------------

void compare_results(const Eigen::VectorXd& eigU,
                     const Eigen::VectorXd& eigT,
                     int m) {
    std::vector<double> U(eigU.data(), eigU.data() + eigU.size());
    std::vector<double> T(eigT.data(), eigT.data() + eigT.size());

    std::sort(U.begin(), U.end());
    std::sort(T.begin(), T.end());

    std::cout << "\nEigenvalues of U:\n";
    for (double x : U) std::cout << x << " ";
    std::cout << "\n\nEigenvalues of T:\n";
    for (double x : T) std::cout << x << " ";
    std::cout << "\n\nPointwise absolute errors:\n";

    for (int i = 0; i < m; ++i)
        std::cout << "i = " << i
                  << " |λ_T - λ_U| = "
                  << std::abs(T[i] - U[i]) << "\n";
}

// ---------------------- Main ----------------------

int main() {
    constexpr MatrixType matrixType = MatrixType::SparseFromFile;

    const int n = 1000;   // matrix size for random case
    const int m = 100;    // Lanczos subspace size
    assert(m <= n);

    Eigen::MatrixXd U;

    if constexpr (matrixType == MatrixType::DenseRandom) {
        std::cout << "Using DENSE RANDOM matrix\n";
        U = Eigen::MatrixXd::Random(n, n);
    }

    else if constexpr (matrixType == MatrixType::DenseFromFile) {
        std::cout << "Using DENSE FROM FILE: " << DENSE_FILE << "\n";
        U = read_dense_input_file(DENSE_FILE);
    }

    else if constexpr (matrixType == MatrixType::SparseFromFile) {
        std::cout << "Using SPARSE FROM FILE: " << "sparse_random_1000.txt" << "\n";
        SpMat U = read_sparse_input_file("/Users/signestjernstoft/CLionProjects/sf2565-project/src/sparse_random_1000.txt");
    }
    Eigen::VectorXd k1 = Eigen::VectorXd::Random(n);
    Eigen::VectorXd q1 = Eigen::VectorXd::Random(n);

    // Lanczos Runtime

    auto t1 = std::chrono::high_resolution_clock::now();
    auto res = lanczos::solve(U, k1, q1, m);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> lanczos_time = t2 - t1;
    std::cout << "\nLanczos time: " << lanczos_time.count() << " s\n";

    // Eigen Runtime

    auto t3 = std::chrono::high_resolution_clock::now();
    Eigen::EigenSolver<Eigen::MatrixXd> esU(U);
    Eigen::VectorXd lambda_U = esU.eigenvalues().real();
    auto t4 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> eigen_time = t4 - t3;
    std::cout << "Eigen full solver time: " << eigen_time.count() << " s\n";

    // Comparison

    Eigen::VectorXd lambda_T = res.eigenvalues.head(m);
    compare_results(lambda_U.head(m), lambda_T, m);

    return 0;
}