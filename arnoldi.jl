using LinearAlgebra, SparseArrays

"""
    Q,H=arnoldi(A,b,m, method)

A simple implementation of the Arnoldi method.
The algorithm will return an Arnoldi "factorization":
Q*H[1:m+1,1:m]-A*Q[:,1:m]=0
where Q is an orthogonal basis of the Krylov subspace
and H a Hessenberg matrix.

The function `my_hw1_gs(Q,w,k)` needs to be available.

Example:
```julia-repl
using Random
A=randn(100,100); b=randn(100);
m=10;
Q,H=arnoldi(A,b,m);
println("1:st should be zero = ", norm(Q*H-A*Q[:,1:m]));
println("2:nd Should be zero = ", norm(Q'*Q-I));
```

"""
function arnoldi_complex(A, b, m, method)
    n=length(b);
    b = b/norm(b)
    Q = zeros(ComplexF64, n, m + 1)  # Initialize Q as a zero matrix with the same element type as A
    H = zeros(ComplexF64, m + 1, m)  # Initialize H as a zero matrix with the same element type as A
    
    println("inited H")
    check_nan_inf(H)
    println("after H check")
    # Ensure b is compatible with Q's type
    #b = Complex{eltype(A)}.(b)
    Q[:,1].=b/norm(b);

    for k=1:m
        w=A*Q[:,k] # Matrix-vector product with last element
        # Orthogonalize w against columns of Q.
        # Implement this function or replace call with code for orthogonalizatio
        h,β,z=method(Q,w,k)
        #check_nan_inf(β)
        if isapprox(β, 0; atol=1e-10)
            print("BETA iS ZERO! for $m")
            break
        end
        #Put Gram-Schmidt coefficients into H
        H[1:(k+1),k]=[h;β]
        check_nan_inf(H)
        # normalize
        Q[:,k+1]=z/β
    end
    return Q,H
end

function shift_inverse_arnoldi(A, b, m, method, shift)
    n=length(b)
    Q = zeros(ComplexF64, n, m + 1)
    H = zeros(ComplexF64, m + 1, m)
    Q[:,1]=b/norm(b);
 
    I_sparse = sparse(I, n, n)

    for k in 1:m
        w = (A-shift*I_sparse) \ Q[:,k]
        h,beta,z = method(Q,w,k)
        H[1:(k+1),k] = [h;beta]
        Q[:,k+1]=z
    end
    return H
end


function arnoldi(A,b,m,method)
    n=length(b)
    Q = zeros(eltype(A), n, m + 1)
    H = zeros(eltype(A),  m + 1, m)
    Q[:,1]=b/norm(b)

    for k=1:m
        w=A*Q[:,k]; # Matrix-vector product with last element
        # Orthogonalize w against columns of Q.
        # method for gram-schmidt: either dgs, tgs, cgs, mgs
        h,β,z=method(Q,w,k);

        #Put Gram-Schmidt coefficients into H
        H[1:(k+1),k]=[h;β];

        # Check for breakdown (β == 0)
        if β ≈ 0
            println("Arnoldi process terminated early at step $k due to zero norm in orthogonalized vector.")
            return Q, H
        end
        # normalize
        Q[:,k+1]=z;
    end
    return Q, H
end

function dgs(Q, w, k)
    Q = Q[:,1:k]
    h = Q'*w
    z = w-Q*h
    g = Q'*z
    z = z-Q*g
    h = h+g
    beta = norm(z)
    z = z/norm(z)

    return h, beta, z
end

function tgs(Q, w, k)
    Q = Q[:,1:k]
    h = Q'*w
    z = w-Q*h

    g = Q'*z
    z = z-Q*g

    f = Q'*z
    z = z-Q*f

    h = h+g+f

    beta = norm(z)
    z = z/norm(z)

    return h, beta, z
end

function cgs(Q, w, k)
    Q = Q[:,1:k]
    h = Q'*w
    z = w-Q*h
    beta = norm(z)
    z = z/norm(z)
    return h, beta, z
end

function mgs(Q, w, k)
    Q = Q[:,1:k]
    # Infer the element type from w, which will match Q’s type
    h = zeros(eltype(w), k)
    z = w
    for i = 1:k
        h[i] = Q[:, i]'*z
        z = z - h[i]*Q[:, i]
    end
    beta = norm(z)
    z = z/beta
    return h, beta, z
end

function check_nan_inf(matrix)
    has_nan = any(isnan, matrix)
    has_inf = any(isinf, matrix)
    
    if has_nan && has_inf
        print("The matrix contains both NaN and Inf values.")
    elseif has_nan
        print("The matrix contains NaN values.")
    elseif has_inf
        print("The matrix contains Inf values.")
    end
end

function testing(method)
    # Define Q with orthonormal columns
    Q = [1.0 0.0; 0.0 1.0; 1.0 1.0] / sqrt(2)  # 3x2 matrix, columns are orthonormal

    # Define a vector w that is not orthogonal to Q's columns
    w = [1.0, 2.0, 3.0]

    # We want to orthogonalize w against the first k columns of Q (k=2)
    h, β, z = method(Q, w, 1)

    # Output the results
    println("Projections h: ", h)
    println("Norm β: ", β)
    println("Orthogonalized vector z: ", z)

    # Verify orthogonality: Q[:, 1:k]' * z should be approximately zero
    println("Check orthogonality with Q's columns: ", Q[:, 1:2]' * z)

end
#testing(mgs)