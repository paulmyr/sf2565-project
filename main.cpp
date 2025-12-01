#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <CLI11.hpp>


void read_input_file(std::string filename) {
    // TODO: everything

    return;
}


int arnoldi_iteration(Eigen::MatrixXd A, int num_eigen_values) {
    // TODO: everything...

    return -1;
}


int main(int argc, char **argv) {
    CLI::App app{"Arnoldi Iteration"};

    std::string input_file;
    app.add_option("--input-file,-i", input_file,
        "Input file path. Expects file in TODO SPECIFY FORMAT");

    CLI11_PARSE(app, argc, argv);

    if (input_file.empty()) {
        Eigen::MatrixXd m(2, 2);
        m(0, 0) = 3;
        m(1, 0) = 2.5;
        m(0, 1) = -1;
        m(1, 1) = m(1, 0) + m(0, 1);

        std::cout << "Example Eigen matrix:\n" << m << std::endl;
    }

    return 0;
}