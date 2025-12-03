#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <CLI11.hpp>
#include "arnoldi.h"


Eigen::MatrixXd read_input_file(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }infile.is_open();

    int rows, cols;
    infile >> rows >> cols;

    Eigen::MatrixXd A(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double value;
            infile >> value;
            A(i, j) = value;
        }
    }

    return A;
}


int main(int argc, char **argv) {
    CLI::App app{"Arnoldi Iteration"};

    std::string input_file;
    app.add_option("--input-file,-i", input_file,
        "Input file path. Expects file in TODO SPECIFY FORMAT");

    CLI11_PARSE(app, argc, argv);

    Eigen::MatrixXd A;
    if (input_file.empty()) {
        A.resize(4,4);
        A << 1, 2, 0, 0,
             2, 3, 1, 0,
             0, 1, 2, 1,
             0, 0, 1, 4;
        std::cout << "Example Eigen matrix:\n" << A << std::endl;
    }
    else {
        A = read_input_file(input_file);
        std::cout << "Eigen matrix:\n" << A << std::endl;
    }

    auto test = arnoldi::solve(A, 3);
    std::cout << test.krylov_basis << std::endl;
    std::cout << test.hessenberg << std::endl;
    std::cout << test.iterations << std::endl;

    return 0;
}