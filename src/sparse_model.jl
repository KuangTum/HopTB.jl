module SparseModel

using LinearAlgebra
using SparseArrays
using StaticArrays

using ..HopTB
using ..HopTB: AbstractTBModel
using ..HopTB.Memoize: @memoize

export SparseTBModel

const RVector = SVector{3,Int16}
const R0 = RVector(0, 0, 0)

mutable struct SparseTBModel{T} <: AbstractTBModel{T}
    norbits::Int64
    lat::SMatrix{3,3,Float64,9}
    rlat::SMatrix{3,3,Float64,9}
    hoppings::Dict{RVector,SparseMatrixCSC{T,Int64}}
    overlaps::Dict{RVector,SparseMatrixCSC{T,Int64}}
    positions::Dict{RVector,SVector{3,SparseMatrixCSC{T,Int64}}}
    isorthogonal::Bool
    nsites::Union{Missing,Int64}
    site_norbits::Union{Missing,Vector{Int16}}
    site_positions::Union{Missing,Matrix{Float64}}
end

function SparseTBModel{T}(norbits::Int64, lat::AbstractMatrix{<:Real};
        isorthogonal::Bool=false) where T
    hoppings = Dict{RVector,SparseMatrixCSC{T,Int64}}()
    overlaps = Dict{RVector,SparseMatrixCSC{T,Int64}}()
    positions = Dict{RVector,SVector{3,SparseMatrixCSC{T,Int64}}}()
    slat = SMatrix{3,3,Float64,9}(lat)
    rlat = 2π * inv(slat)'
    return SparseTBModel{T}(norbits, slat, rlat, hoppings, overlaps, positions,
        isorthogonal, missing, missing, missing)
end

SparseTBModel(norbits::Int64, lat::AbstractMatrix{<:Real};
        isorthogonal::Bool=false) = SparseTBModel{ComplexF64}(norbits, lat; isorthogonal=isorthogonal)

function _ensure_matrix_block!(dict::Dict{RVector,SparseMatrixCSC{T,Int64}}, R::AbstractVector{<:Integer},
        size::Int) where T
    key = RVector(Int16.(R))
    if !haskey(dict, key)
        dict[key] = spzeros(T, size, size)
    end
    return key
end

function _ensure_position_block!(dict::Dict{RVector,SVector{3,SparseMatrixCSC{T,Int64}}},
        R::AbstractVector{<:Integer}, size::Int) where T
    key = RVector(Int16.(R))
    if !haskey(dict, key)
        dict[key] = SVector{3,SparseMatrixCSC{T,Int64}}(
            spzeros(T, size, size),
            spzeros(T, size, size),
            spzeros(T, size, size))
    end
    return key
end

function HopTB.sethopping!(tm::SparseTBModel{T}, R::AbstractVector{<:Integer},
        i::Int, j::Int, value::Number) where T
    size = tm.norbits
    keyR = _ensure_matrix_block!(tm.hoppings, R, size)
    keyZero = R0
    val = T(value)
    keyNeg = _ensure_matrix_block!(tm.hoppings, -R, size)
    if keyR == keyZero && i == j
        tm.hoppings[keyR][i, j] = real(val)
    else
        tm.hoppings[keyR][i, j] = val
        tm.hoppings[keyNeg][j, i] = conj(val)
    end
    return nothing
end

function HopTB.setoverlap!(tm::SparseTBModel{T}, R::AbstractVector{<:Integer},
        i::Int, j::Int, value::Number) where T
    tm.isorthogonal && error("tm is orthogonal.")
    size = tm.norbits
    keyR = _ensure_matrix_block!(tm.overlaps, R, size)
    keyZero = R0
    val = T(value)
    keyNeg = _ensure_matrix_block!(tm.overlaps, -R, size)
    if keyR == keyZero && i == j
        tm.overlaps[keyR][i, j] = real(val)
    else
        tm.overlaps[keyR][i, j] = val
        tm.overlaps[keyNeg][j, i] = conj(val)
    end
    return nothing
end

function HopTB.setposition!(tm::SparseTBModel{T}, R::AbstractVector{<:Integer},
        i::Int, j::Int, α::Int, value::Number) where T
    size = tm.norbits
    keyR = _ensure_position_block!(tm.positions, R, size)
    keyZero = R0
    val = T(value)
    keyNeg = _ensure_position_block!(tm.positions, -R, size)
    if keyR == keyZero && i == j
        tm.positions[keyR][α][i, j] = real(val)
    else
        tm.positions[keyR][α][i, j] = val
        keyNegOverlap = RVector(Int16.(-R))
        tmp = haskey(tm.overlaps, keyNegOverlap) ? tm.overlaps[keyNegOverlap][j, i] : zero(T)
        shift = (tm.lat * Float64.(R))[α]
        tm.positions[keyNeg][α][j, i] = conj(val) - shift * tmp
    end
    return nothing
end

function HopTB.change_energy_reference(tm::SparseTBModel, μ::Number)
    ntm = deepcopy(tm)
    key0 = R0
    if ntm.isorthogonal
        diag = spdiagm(0 => fill(ComplexF64(1.0), tm.norbits))
        base = get(ntm.hoppings, key0, spzeros(ComplexF64, tm.norbits, tm.norbits))
        ntm.hoppings[key0] = base - μ * diag
    else
        for (R, hopping) in tm.hoppings
            overlap = get(ntm.overlaps, R, spzeros(ComplexF64, tm.norbits, tm.norbits))
            ntm.hoppings[R] = hopping - μ * overlap
        end
    end
    return ntm
end

function _accumulate_sparse!(dest::SparseMatrixCSC{T,Int32}, src::SparseMatrixCSC{T,Int32}, coeff::T) where T
    nnz(src) == 0 && return dest
    if coeff == zero(T)
        return dest
    end
    if nnz(dest) == 0
        dest .= coeff .* src
    else
        dest .+= coeff .* src
    end
    return dest
end

function _new_sparse(n::Int, T::Type{<:Number})
    return spzeros(T, n, n)
end

end
