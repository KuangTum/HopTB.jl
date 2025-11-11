using SparseArrays
using .SparseModel
using .SparseModel: SparseTBModel
using ..HopTB.Memoize: @memoize

@memoize k function getdH(tm::SparseTBModel{T}, order::Tuple{Int64,Int64,Int64},
        k::AbstractVector{<:Real}) where T
    n = tm.norbits
    total_order = sum(order)
    result = spzeros(ComplexF64, n, n)
    for (R, block) in tm.hoppings
        Rvec = Float64.(R)
        coeff = exp(im * 2π * dot(k, Rvec))
        if total_order > 0
            Rc = tm.lat * Rvec
            coeff *= (im * Rc[1])^order[1] * (im * Rc[2])^order[2] * (im * Rc[3])^order[3]
        end
        nnz(block) == 0 && continue
        result += coeff .* block
    end
    if order == (0, 0, 0)
        return (result + result') / 2
    else
        return result
    end
end

@memoize k function getdS(tm::SparseTBModel{T}, order::Tuple{Int64,Int64,Int64},
        k::AbstractVector{<:Real}) where T
    n = tm.norbits
    if tm.isorthogonal
        return order == (0,0,0) ? spdiagm(0 => fill(ComplexF64(1.0), n)) : spzeros(ComplexF64, n, n)
    end
    result = spzeros(ComplexF64, n, n)
    total_order = sum(order)
    for (R, block) in tm.overlaps
        Rvec = Float64.(R)
        coeff = exp(im * 2π * dot(k, Rvec))
        if total_order > 0
            Rc = tm.lat * Rvec
            coeff *= (im * Rc[1])^order[1] * (im * Rc[2])^order[2] * (im * Rc[3])^order[3]
        end
        result += coeff .* block
    end
    if order == (0, 0, 0)
        return (result + result') / 2
    else
        return result
    end
end

@memoize k function getdAw(tm::SparseTBModel{T}, α::Int64, order::Tuple{Int64,Int64,Int64},
        k::AbstractVector{<:Real}) where T
    n = tm.norbits
    result = SparseModel._new_sparse(n, Complex{Float64})
    total_order = sum(order)
    for (R, blocks) in tm.positions
        Rvec = Float64.(R)
        coeff = exp(im * 2π * dot(k, Rvec))
        if total_order > 0
            Rc = tm.lat * Rvec
            coeff *= (im * Rc[1])^order[1] * (im * Rc[2])^order[2] * (im * Rc[3])^order[3]
        end
        result += coeff .* blocks[α]
    end
    return transpose(result)
end
