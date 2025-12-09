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

using SpMat   = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

enum class MatrixType {
    DenseRandom,
    DenseFromFile,
    SparseFromFile
};

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

int main() {
    constexpr MatrixType matrixType = MatrixType::SparseFromFile;

    const int n = 10;   // matrix size for random case
    const int m = 8;    // Lanczos subspace size
    assert(m <= n);

    /*
    // Dense rando, matrix
    Eigen::MatrixXd U;
    std::cout << "Using DENSE RANDOM matrix\n";
    U = Eigen::MatrixXd::Random(n, n);
    */

    /*
    // Dense matrix from file
    Eigen::MatrixXd U;
    std::cout << "Using DENSE FROM FILE: " << "densefile.." << "\n";
    U = read_dense_input_file("densefile");
    */

    SpMat U = read_sparse_input_file("/Users/signestjernstoft/CLionProjects/sf2565-project/src/sparse_random_10.txt");
    const int actual_n = U.rows();
    std::cout << "Dimension of sparse matrix: " << actual_n << '\n';

    Eigen::VectorXd k1 = Eigen::VectorXd::Random(n);
    Eigen::VectorXd q1 = Eigen::VectorXd::Random(n);

    // Lanczos Runtime
    auto t1 = std::chrono::high_resolution_clock::now();
    auto res = lanczos::solve(U, k1, q1, m);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> lanczos_time = t2 - t1;
    std::cout << "\nLanczos time: " << lanczos_time.count() << " s\n";

    // Eigen Runtime

    Eigen::MatrixXd U_dense = Eigen::MatrixXd(U); // only necessary if sparse
    auto t3 = std::chrono::high_resolution_clock::now();
    Eigen::EigenSolver<Eigen::MatrixXd> esU(U_dense);
    Eigen::VectorXd lambda_U = esU.eigenvalues().real();
    auto t4 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> eigen_time = t4 - t3;
    std::cout << "Eigen full solver time: " << eigen_time.count() << " s\n";

    // Comparison
    Eigen::VectorXd lambda_T = res.eigenvalues.head(m);
    compare_results(lambda_U.head(m), lambda_T, m);

    return 0;
}