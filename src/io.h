//
// Created by Paul Mayer on 09.12.25.
//

#ifndef SF2565_PROJECT_IO_H
#define SF2565_PROJECT_IO_H

#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <fstream>
#include <stdexcept>

using SparseMatrixXd = Eigen::SparseMatrix<double>;
using MatrixXd = Eigen::MatrixXd;
using Triplet = Eigen::Triplet<double>;

template<typename TMatrix>
TMatrix read_input_file(const std::string& filename);

template<>
MatrixXd read_input_file<MatrixXd>(const std::string& filename) {
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
    MatrixXd A(n, n);
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


template<>
SparseMatrixXd read_input_file<SparseMatrixXd>(const std::string& filename) {
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

    SparseMatrixXd A(rows, cols);
    A.setFromTriplets(coefficients.begin(), coefficients.end());

    return A;
}

#endif //SF2565_PROJECT_IO_H