//
// Created by Paul Mayer on 02.12.25.
//

#ifndef SF2565_PROJECT_ARNOLDI_H
#define SF2565_PROJECT_ARNOLDI_H

#include <assert.h>
#include <Eigen/Dense>

/*
 * Reference implementation in Julia by Signe Stjernstoft.
 *
 * function arnoldi(A,b,m,method)
 *     n=length(b)
 *     Q = zeros(eltype(A), n, m + 1)
 *     H = zeros(eltype(A),  m + 1, m)
 *     Q[:,1]=b/norm(b)
 *
 *     for k=1:m
 *         w=A*Q[:,k]; # Matrix-vector product with last element
 *         # Orthogonalize w against columns of Q.
 *         # method for gram-schmidt: either dgs, tgs, cgs, mgs
 *         h,β,z=method(Q,w,k);
 *
 *         #Put Gram-Schmidt coefficients into H
 *         H[1:(k+1),k]=[h;β];
 *
 *         # Check for breakdown (β == 0)
 *         if β ≈ 0
 *             println("Arnoldi process terminated early at step $k due to zero norm in orthogonalized vector.")
 *             return Q, H
 *         end
 *         # normalize
 *         Q[:,k+1]=z;
 *     end
 *     return Q, H
 * end
 *
 */


namespace {
    template <typename TVector>
    struct GramSchmidtResult {
        TVector h;
        double beta;
        TVector z;
    };

    GramSchmidtResult<Eigen::VectorXd> gram_schmidt(
        const Eigen::MatrixXd &Q,
        Eigen::VectorXd w,
        int k)
    {
        Eigen::VectorXd h(k+1);
        for (int i = 0; i <= k; i++) {
            h(i) = Q.col(i).dot(w);
            w = w - h(i) * Q.col(i);
        }
        return {h, w.norm(), w / w.norm()};
    }

    GramSchmidtResult<Eigen::VectorXd> c_gram_schmidt() {
        throw std::logic_error("Function not yet implemented");
    }

    GramSchmidtResult<Eigen::VectorXd> d_gram_schmidt() {
        throw std::logic_error("Function not yet implemented");
    }

    GramSchmidtResult<Eigen::VectorXd> t_gram_schmidt() {
        throw std::logic_error("Function not yet implemented");
    }

    GramSchmidtResult<Eigen::VectorXd> m_gram_schmidt() {
        throw std::logic_error("Function not yet implemented");
    }

} // namespace

namespace arnoldi {

    /*
     * Todo: actually compute the eigenvalues and vectors using Arnoldi.
     * Not really sure how that works yet though...
     */

    template <typename TMatrix>
    struct ArnoldiResult {
        //TVector eigenvalues;
        //TMatrix eigenvectors;
        TMatrix krylov_basis;
        TMatrix hessenberg;
        int iterations;
    };


    /**
     * todo: write doc
     *
     * @param A todo
     * @param num_eigenvalues todo
     * @param tol todo
     * @return
     */
    ArnoldiResult<Eigen::MatrixXd> solve(const Eigen::MatrixXd &A, int num_eigenvalues, const double tol = 1e-8) {
        assert(A.rows() == A.cols());
        assert(num_eigenvalues < A.rows());

        const int m = num_eigenvalues;
        const int n = A.cols();
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, m+1);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m+1, m);
        Eigen::VectorXd b = Eigen::VectorXd::Random(n);

        Q.col(0) = b.normalized();

        Eigen::VectorXd w(n);

        int k = 0;
        for (;k < m; ++k) {
            w = A * Q.col(k);

            // h,β,z=method(Q,w,k);
            auto gs = gram_schmidt(Q, w, k);

            H.col(k).head(k+1) = gs.h;
            H(k+1, k) = gs.beta;

            // H[1:(k+1),k]=[h;β];
            if (gs.beta < tol) {
                break;
            }

            Q.col(k+1) = gs.z;
        }

        return ArnoldiResult<Eigen::MatrixXd> {
            Q.leftCols(k+1),
            H.topLeftCorner(k+1, k),
            k
        };
    }

}

#endif //SF2565_PROJECT_ARNOLDI_H