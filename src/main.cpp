#include <CLI11.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <vector>

#include "io.h"
#include "lanczos.h"

void simon_test() {
  // Test for Lanczos
  std::cout << "Simon Test for new Lanczos" << std::endl;

  constexpr int n = 4;
  Eigen::MatrixXd U(n, n);
  U << 1, 2, 0, 0, 2, 3, 1, 0, 0, 1, 2, 1, 0, 0, 1, 4;

  std::cout << "U =\n" << U << "\n\n";

  Eigen::VectorXd k1 = Eigen::VectorXd::Random(n);
  Eigen::VectorXd q1 = Eigen::VectorXd::Random(n);

  constexpr int m = 4;
  const auto res = lanczos::solve(U, k1, q1, m);

  std::cout << "T (tridiagonal reduction) =\n" << res.T << "\n\n";

  // Eigenvalues of U
  Eigen::EigenSolver<Eigen::MatrixXd> esU(U);
  Eigen::VectorXd lambda_U = esU.eigenvalues().real();

  // Eigenvalues of T (already computed in res.eigenvalues)
  Eigen::VectorXd lambda_T = res.eigenvalues;

  std::vector<double> eigU(lambda_U.data(), lambda_U.data() + lambda_U.size());
  std::vector<double> eigT(lambda_T.data(), lambda_T.data() + lambda_T.size());
  std::ranges::sort(eigU);
  std::ranges::sort(eigT);

  std::cout << "Eigenvalues of U (sorted): ";
  for (const double x : eigU)
    std::cout << x << " ";
  std::cout << "\nEigenvalues of T (sorted): ";
  for (const double x : eigT)
    std::cout << x << " ";
  std::cout << "\n";

  std::cout << "Pointwise absolute errors:\n";
  for (int i = 0; i < eigU.size(); ++i) {
    const double err = std::abs(eigT[i] - eigU[i]);
    std::cout << "i = " << i << ": |λ_T - λ_U| = " << err << "\n";
  }
}

void read_and_solve(const std::string &input_file, const int m = 8) {
  // Read Sparse Matrix:
  lanczos::SparseMatrixXd U;

  try {
    U = lanczos::io::read_input_file<lanczos::SparseMatrixXd>(input_file);
    std::cout << "Loaded sparse matrix: " << U.rows() << "x" << U.cols()
              << " with " << U.nonZeros() << " non-zeros.\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return;
  }

  Eigen::VectorXd k0 = Eigen::VectorXd::Ones(U.rows());
  Eigen::VectorXd q0 = Eigen::VectorXd::Ones(U.rows());

  // Lanczos Runtime
  auto t_lanczos_start = std::chrono::high_resolution_clock::now();
  auto result = lanczos::solve(U, k0, q0, m);
  auto t_lanczos_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> lanczos_time = t_lanczos_end - t_lanczos_start;
  std::cout << "\nLanczos time: " << lanczos_time.count() << " s\n";

  // Results from lanczos
  std::cout << "Tridiagonal T:\n" << result.T << "\n\n";
  std::cout << "Eigenvalue approximations:\n"
            << result.eigenvalues.transpose() << "\n\n";
  std::cout << "Iterations: " << result.iterations << "\n";

  // True eigs
  if (U.rows() <= 100) {
    Eigen::MatrixXd U_dense = U;
    Eigen::EigenSolver<Eigen::MatrixXd> full(U_dense);
    std::cout << "\nTrue eigenvalues of U:\n"
              << full.eigenvalues().transpose() << "\n";
  }
}

int main(int argc, char **argv) {
  CLI::App app{"Lanczos Solver"};

  std::string input_file;
  app.add_option("--input-file,-i", input_file,
                 "Input file path. Expects matrix in COO (also known as ija or "
                 "triplet) file format");

  int num_eigenvalues;
  app.add_option("--num-eigenvalues, -n", num_eigenvalues,
                 "Sets the m variable, default: 5");

  if (num_eigenvalues == 0)
    num_eigenvalues = 5;

  CLI11_PARSE(app, argc, argv);

  std::cout << "File: " << input_file << std::endl;
  if (input_file != "") {
    read_and_solve(input_file, num_eigenvalues);
  }

  return 0;
}
