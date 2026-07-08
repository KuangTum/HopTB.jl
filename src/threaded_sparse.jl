# Threaded kernels for SparseMatrixCSC products. Julia's SparseArrays kernels
# are single-threaded; at ~1e8 nnz they stream ~16 GB/s on nodes with >300 GB/s
# of bandwidth, which makes sparse*dense the bound of the orthonormalization
# and velocity stages. Threading is over independent output columns.

using SparseArrays
using LinearAlgebra

"""
    spmm_threaded(A::SparseMatrixCSC, B::AbstractMatrix)

`A * B` threaded over columns of `B`. Each column uses the standard serial
SpMV kernel, so the result is bitwise identical to `A * B`.
"""
function spmm_threaded(A::SparseMatrixCSC{Tv}, B::AbstractMatrix{Tb}) where {Tv,Tb}
    T = promote_type(Tv, Tb)
    C = zeros(T, size(A, 1), size(B, 2))
    return spmm_threaded!(C, A, B)
end

function spmm_threaded!(C::AbstractMatrix, A::SparseMatrixCSC, B::AbstractMatrix)
    (size(A, 2) == size(B, 1) && size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2)) ||
        throw(DimensionMismatch("spmm_threaded!: got $(size(C)) = $(size(A)) * $(size(B))"))
    if Threads.nthreads() == 1
        return mul!(C, A, B)
    end
    Threads.@threads for j in 1:size(B, 2)
        @views mul!(C[:, j], A, B[:, j])
    end
    return C
end

# Dispatch helper: threaded product when sparse, plain product otherwise.
_sparse_times_dense(M::SparseMatrixCSC, X::AbstractMatrix) = spmm_threaded(M, X)
_sparse_times_dense(M::AbstractMatrix, X::AbstractMatrix) = M * X

"""
    hermitian_spmv_threaded!(y, A, x)

`y = A * x` for a **Hermitian** sparse `A`, evaluated as `y[j] = conj(A[:,j])⋅x`
(valid because `Aᴴ = A`). Column dot products write disjoint entries of `y`,
so this threads without scatter races. Caller must guarantee hermiticity.
"""
function hermitian_spmv_threaded!(y::AbstractVector{T}, A::SparseMatrixCSC{T},
        x::AbstractVector{T}) where {T<:Complex}
    n = size(A, 2)
    (length(y) == n && length(x) == size(A, 1)) ||
        throw(DimensionMismatch("hermitian_spmv_threaded!"))
    colptr = A.colptr; rowval = A.rowval; nzval = A.nzval
    Threads.@threads for j in 1:n
        acc = zero(T)
        @inbounds for p in colptr[j]:(colptr[j+1] - 1)
            acc += conj(nzval[p]) * x[rowval[p]]
        end
        @inbounds y[j] = acc
    end
    return y
end
