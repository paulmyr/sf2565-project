import random
import numpy as np
from scipy import sparse

def generate_sparse_matrix(n, density):
    with open(f"random_matrix_{n}.txt", "w") as f:
        f.write(f"{n}\n")
        
        for _ in range(n):
            row_values = []
            for _ in range(n):
                if random.random() < density:
                    val = random.randint(0, 9)
                    row_values.append(f"{val}")
                else:
                    row_values.append("0")
            
            f.write(" ".join(row_values) + "\n")


def generate_dense_convection_diffusion(matrix_size, asymmetry_delta, filename):
    main_diag = 2.0 * np.ones(matrix_size)
    off_diag_down = (-1.0 - asymmetry_delta) * np.ones(matrix_size - 1)
    off_diag_up = (-1.0 + asymmetry_delta) * np.ones(matrix_size - 1)

    matrix = np.diag(main_diag) + \
             np.diag(off_diag_down, k=-1) + \
             np.diag(off_diag_up, k=1)

    with open(filename, 'w') as f:
        f.write(f"{matrix_size}\n")
        np.savetxt(f, matrix, fmt='%.2f')


def generate_sparse_convection_diffusion(matrix_size, asymmetry_delta, filename):
    diagonals = [
        2.0 * np.ones(matrix_size),
        (-1.0 - asymmetry_delta) * np.ones(matrix_size - 1),
        (-1.0 + asymmetry_delta) * np.ones(matrix_size - 1)
    ]
    offsets = [0, -1, 1]
    
    matrix = sparse.diags(diagonals, offsets, shape=(matrix_size, matrix_size)).tocoo()

    with open(filename, 'w') as f:
        f.write(f"{matrix_size} {matrix_size} {matrix.nnz}\n")
        for row, col, val in zip(matrix.row, matrix.col, matrix.data):
            f.write(f"{row} {col} {val:.2f}\n")


def generate_sparse_random(matrix_size, filename):
    diagonals = [np.random.rand(matrix_size),
                 np.random.rand(matrix_size-1),
                 np.random.rand(matrix_size-1)
    ]
    offsets = [0, -1, 1]
    
    matrix = sparse.diags(diagonals, offsets, shape=(matrix_size, matrix_size)).tocoo()

    with open(filename, 'w') as f:
        f.write(f"{matrix_size} {matrix_size} {matrix.nnz}\n")
        for row, col, val in zip(matrix.row, matrix.col, matrix.data):
            f.write(f"{row} {col} {val:.4f}\n")


if __name__ == "__main__":
    #generate_sparse_matrix(10, 0.5)
    #generate_dense_convection_diffusion(1000, 0, "dense_convection_diffusion_1000.txt")
    #generate_sparse_convection_diffusion(100000, 0, "sparse_convection_diffusion_100000.txt")
    generate_sparse_random(1000, "../../assignment4/sparse_random_1000.txt")