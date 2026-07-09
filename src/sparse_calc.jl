module SparseCalc

using LinearAlgebra
using SparseArrays
using Random
using Arpack
using HDF5

using ..HopTB
using ..HopTB: AbstractTBModel
using ..HopTB.Memoize: @memoize

export PartialHermEig, eigs_near, eigs_near_block, eigs_window, pardiso_available

# ----- HDF5 eigenpair cache (full-diag only) -------------------------------
# Set ENV["HOPTB_EIGCACHE_DIR"] to a directory before the first eigs_near
# call. When this is set and `_dense_eigs` is invoked, the (energies, vectors)
# from a full diagonalization are stashed to/loaded from
#   <dir>/eig_n<N>_k<k1>_<k2>_<k3>_r<ref>_w<win>.h5
# so a subsequent run at the same (k, reference, window) reads instead of
# diagonalizing. Use a snapshot-specific dir to avoid cross-model collisions.
function _eigcache_dir()
    val = get(ENV, "HOPTB_EIGCACHE_DIR", "")
    return isempty(strip(val)) ? nothing : strip(val)
end

function _eigcache_filename(dir::AbstractString, n_orb::Integer,
        k::AbstractVector{<:Real}, ref::Float64, window::Union{Nothing,Real})
    win = window === nothing ? Inf : Float64(window)
    _f(x) = string(round(Float64(x); digits=6))
    fname = string("eig_n", Int(n_orb),
                   "_k", _f(k[1]), "_", _f(k[2]), "_", _f(k[3]),
                   "_r", _f(ref),
                   "_w", isfinite(win) ? _f(win) : "inf",
                   ".h5")
    return joinpath(dir, fname)
end

function _eigcache_fingerprint(tm::AbstractTBModel)::Float64
    # Cheap "is this still the same model?" check. Sums the lattice-vector
    # diagonal — robust enough that a different snapshot (geometry change)
    # produces a different fingerprint, while the same model rebuilt
    # deterministically reproduces it.
    L = tm.lat
    return Float64(L[1,1] + L[2,2] + L[3,3])
end

function _eigcache_load(path::AbstractString, tm::AbstractTBModel,
        k::AbstractVector{<:Real}, ref::Float64,
        window::Union{Nothing,Real})
    isfile(path) || return nothing
    try
        h5open(path, "r") do f
            n_orb = read(f, "n_orbits")
            n_orb == tm.norbits || return nothing
            fp = read(f, "lat_fingerprint")
            isapprox(fp, _eigcache_fingerprint(tm); atol=1e-9, rtol=1e-9) || return nothing
            energies = read(f, "energies")::Vector{Float64}
            vectors  = read(f, "vectors")::Matrix{ComplexF64}
            residuals = haskey(f, "residuals") ? read(f, "residuals")::Vector{Float64} :
                zeros(Float64, length(energies))
            return PartialHermEig(energies, vectors, residuals, ref)
        end
    catch err
        @warn "Eigcache read failed; falling back to diagonalization." path exception=err
        return nothing
    end
end

function _eigcache_save(path::AbstractString, tm::AbstractTBModel,
        k::AbstractVector{<:Real}, ref::Float64, window::Union{Nothing,Real},
        spec)
    try
        mkpath(dirname(path))
        win = window === nothing ? Inf : Float64(window)
        h5open(path, "w") do f
            write(f, "version",   1)
            write(f, "n_orbits",  Int(tm.norbits))
            write(f, "lat_fingerprint", _eigcache_fingerprint(tm))
            write(f, "reference", ref)
            write(f, "window",    win)
            write(f, "k",         Float64.(collect(k)))
            write(f, "energies",  spec.values)
            write(f, "vectors",   spec.vectors)
            write(f, "residuals", spec.residuals)
        end
    catch err
        @warn "Eigcache write failed." path exception=err
    end
    return nothing
end

const DEFAULT_DENSE_THRESHOLD = 64
const DEFAULT_PARDISO_MIN_SIZE = 96
const DEFAULT_SOLVER = :pardiso
const DEFAULT_ILL_THRESHOLD = 5e-4

const _pardiso_supported = Ref(true)
const _pardiso_available = Ref(false)
const _pardiso_init_error = Ref{Union{Nothing,Exception}}(nothing)

let pkg_path = Base.find_package("Pardiso")
    if pkg_path === nothing
        _pardiso_supported[] = false
        _pardiso_available[] = false
    else
        try
            @eval import Pardiso
            _pardiso_available[] = Pardiso.mkl_is_available()
        catch err
            _pardiso_supported[] = false
            _pardiso_available[] = false
            _pardiso_init_error[] = err
        end
    end
end

_pardiso_active() = _pardiso_supported[] && _pardiso_available[]

function pardiso_available()
    return _pardiso_active()
end

function _ill_threshold()
    val = get(ENV, "HOPTB_ILL_THRESHOLD", nothing)
    if val === nothing || isempty(strip(val))
        return DEFAULT_ILL_THRESHOLD
    end
    try
        return parse(Float64, strip(val))
    catch
        return DEFAULT_ILL_THRESHOLD
    end
end

struct PartialHermEig
    values::Vector{Float64}
    vectors::Matrix{ComplexF64}
    residuals::Vector{Float64}
    reference::Float64
end

Base.length(spec::PartialHermEig) = length(spec.values)
Base.size(spec::PartialHermEig) = (size(spec.vectors, 1), length(spec))

function Base.iterate(spec::PartialHermEig)
    return (spec.values, Val(:vectors))
end

function Base.iterate(spec::PartialHermEig, ::Val{:vectors})
    return (spec.vectors, Val(:residuals))
end

function Base.iterate(spec::PartialHermEig, ::Val{:residuals})
    return (spec.residuals, Val(:reference))
end

function Base.iterate(spec::PartialHermEig, ::Val{:reference})
    return (spec.reference, Val(:done))
end

Base.iterate(::PartialHermEig, ::Val{:done}) = nothing

function PartialHermEig(values::Vector{Float64}, vectors::Matrix{ComplexF64},
    residuals::Vector{Float64}, reference::Real)
    nstates = length(values)
    nstates == size(vectors, 2) || throw(ArgumentError("vectors must have one column per eigenvalue."))
    nstates == length(residuals) ||
        throw(ArgumentError("residuals must have the same length as values."))
    return PartialHermEig(values, vectors, residuals, Float64(reference))
end

function _dense_eigs(tm::AbstractTBModel, k::AbstractVector{<:Real},
    reference::Float64, window::Union{Nothing,Real})::PartialHermEig
    cache_dir = _eigcache_dir()
    cache_path = cache_dir === nothing ? nothing :
        _eigcache_filename(cache_dir, tm.norbits, k, reference, window)
    if cache_path !== nothing
        cached = _eigcache_load(cache_path, tm, k, reference, window)
        if cached !== nothing
            return cached
        end
    end
    spec = HopTB.geteig(tm, k)
    energies = Float64.(spec.values)
    order = sortperm(abs.(energies .- reference))
    energies = energies[order]
    vectors = spec.vectors[:, order]
    residuals = zeros(Float64, length(energies))
    if window !== nothing
        win = Float64(window)
        mask = abs.(energies .- reference) .<= win
        energies = energies[mask]
        vectors = vectors[:, mask]
        residuals = residuals[mask]
    end
    result = PartialHermEig(energies, vectors, residuals, reference)
    if cache_path !== nothing
        _eigcache_save(cache_path, tm, k, reference, window, result)
    end
    return result
end

function _orthonormalize_eigenpairs(H::AbstractMatrix{ComplexF64},
        S::Union{Nothing,AbstractMatrix{ComplexF64}},
        energies::Vector{Float64},
        vecs::Matrix{ComplexF64},
        ill_threshold::Float64)

    S === nothing && return energies, vecs
    m = size(vecs, 2)
    m == 0 && return energies, vecs

    # Step 1: Euclidean QR to get subspace basis
    Q = Matrix(qr(vecs).Q)                    # n×m, Q'Q = I

    # Step 2: subspace matrices.
    # H, S may be sparse (the SparseTBModel path) — Q' * sparse * Q produces
    # a small dense m×m matrix without ever materializing dense H/S.
    # Same parenthesization trick as in getvelocity_formula: `Q' * (M * Q)` runs
    # SparseMatrixCSC * Matrix first (optimized), avoiding the slow dense*sparse path.
    Ssub = Hermitian(Q' * HopTB._sparse_times_dense(S, Q))
    Hsub = Hermitian(Q' * HopTB._sparse_times_dense(H, Q))

    # Step 3: eigen-decompose Ssub and drop ill-conditioned directions
    evalS, vecS = eigen(Ssub)
    keep = abs.(evalS) .> ill_threshold
    if !all(keep)
        @warn "ill-conditioned eigenvalues detected, projected out $(count(!, keep)) eigenvalues"
    end
    V = vecS[:, keep]
    Λ = evalS[keep]
    if length(Λ) == 0
        return energies, vecs
    end

    # Step 4: reduced generalized problem
    Qp = Q * V
    Ssub_p = Matrix(Diagonal(real.(Λ)))
    Hsub_p = Hermitian(Qp' * HopTB._sparse_times_dense(H, Qp))
    eval_new, Z = eigen(Hsub_p, Hermitian(Ssub_p))

    # Step 5: map back
    X = Qp * Z
    return Float64.(eval_new), X
end

function _compute_residuals(H::AbstractMatrix{<:Complex}, S::Union{Nothing,AbstractMatrix{<:Complex}},
    eigenvals::Vector{Float64}, eigenvecs::Matrix{ComplexF64})::Vector{Float64}
    nstates = length(eigenvals)
    residuals = similar(eigenvals)
    if S === nothing
        for (icol, λ) in enumerate(eigenvals)
            residuals[icol] = norm(H * view(eigenvecs, :, icol) - λ * view(eigenvecs, :, icol))
        end
    else
        for (icol, λ) in enumerate(eigenvals)
            residuals[icol] = norm(H * view(eigenvecs, :, icol) - λ * (S * view(eigenvecs, :, icol)))
        end
    end
    return residuals
end

_pardiso_matrixtype(::Type{<:Real}) = Pardiso.REAL_SYM_INDEF
_pardiso_matrixtype(::Type{<:Complex}) = Pardiso.COMPLEX_HERM_INDEF

struct PardisoShiftOp
    solver
    matrix
    rhs_buffer::Vector{ComplexF64}
    S::Union{Nothing,AbstractMatrix{ComplexF64}}
    S_herm::Bool     # S verified Hermitian -> threaded conj-column SpMV is valid
    dummy::Vector{ComplexF64}
    n::Int
end

# Wall-time split of the shift-invert matvec (S-multiply vs Pardiso solve),
# accumulated across all mul! calls in this process. Read/reset from drivers.
const MATVEC_STATS = Dict{Symbol,Float64}(:n => 0.0, :smul => 0.0, :solve => 0.0)

function reset_matvec_stats!()
    for key in keys(MATVEC_STATS)
        MATVEC_STATS[key] = 0.0
    end
    return nothing
end

matvec_stats() = copy(MATVEC_STATS)

_pardiso_cleanup!(op::PardisoShiftOp) = begin
    Pardiso.set_phase!(op.solver, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(op.solver, op.matrix, op.dummy)
    return nothing
end

function Base.size(op::PardisoShiftOp)
    return (op.n, op.n)
end

function Base.size(op::PardisoShiftOp, dim::Int)
    if dim == 1 || dim == 2
        return op.n
    end
    return 1
end

Base.eltype(::Type{PardisoShiftOp}) = ComplexF64

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, op::PardisoShiftOp, x::AbstractVector{ComplexF64})
    t0 = time_ns()
    if op.S === nothing
        copyto!(op.rhs_buffer, x)
    elseif op.S_herm && op.S isa SparseMatrixCSC{ComplexF64,Int64} && Threads.nthreads() > 1
        HopTB.hermitian_spmv_threaded!(op.rhs_buffer, op.S, x)
    else
        mul!(op.rhs_buffer, op.S, x)
    end
    t1 = time_ns()
    Pardiso.set_phase!(op.solver, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(op.solver, y, op.matrix, op.rhs_buffer)
    t2 = time_ns()
    MATVEC_STATS[:n] += 1
    MATVEC_STATS[:smul] += (t1 - t0) / 1e9
    MATVEC_STATS[:solve] += (t2 - t1) / 1e9
    return y
end

function Base.:*(op::PardisoShiftOp, x::AbstractVector{ComplexF64})
    y = similar(x)
    mul!(y, op, x)
    return y
end

function _to_sparse(mat)
    if mat isa SparseMatrixCSC
        return copy(mat)
    else
        return sparse(mat)
    end
end

function _build_pardiso_linear_operator(H::AbstractMatrix{ComplexF64},
    S::Union{Nothing,AbstractMatrix{ComplexF64}}, sigma::Float64)::PardisoShiftOp
    n = size(H, 1)
    Hsp = _to_sparse(H)
    if S === nothing
        shift = copy(Hsp)
        shift .-= sigma * spdiagm(0 => fill(ComplexF64(1.0), n))
    else
        Ssp = _to_sparse(S)
        shift = Hsp - sigma * Ssp
    end
    A = shift
    solver = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(solver, _pardiso_matrixtype(eltype(A)))
    Pardiso.pardisoinit(solver)
    Pardiso.fix_iparm!(solver, :N)
    Ap = Pardiso.get_matrix(solver, A, :N)
    dummy = zeros(ComplexF64, n)
    Pardiso.set_phase!(solver, Pardiso.ANALYSIS)
    Pardiso.pardiso(solver, Ap, dummy)
    Pardiso.set_phase!(solver, Pardiso.NUM_FACT)
    Pardiso.pardiso(solver, Ap, dummy)
    rhs_buffer = zeros(ComplexF64, n)
    # Store S as passed — sparse stays sparse. (The field was previously typed
    # Union{Nothing,Matrix{ComplexF64}}, which silently densified a sparse S:
    # a 193 GB allocation at 110k orbits, re-streamed by every ARPACK matvec.)
    # HOPTB_FORCE_DENSE_S=true restores the old behavior for A/B benchmarks.
    S_store = (S !== nothing && get(ENV, "HOPTB_FORCE_DENSE_S", "") == "true") ?
        Matrix{ComplexF64}(S) : S
    S_herm = S_store isa SparseMatrixCSC{ComplexF64,Int64} && ishermitian(S_store)
    return PardisoShiftOp(solver, Ap, rhs_buffer, S_store, S_herm, dummy, n)
end

function _arpack_eigs_pardiso(H::AbstractMatrix{ComplexF64},
    S::Union{Nothing,AbstractMatrix{ComplexF64}}, nev::Int,
    sigma::Float64, ncv::Int, tol::Float64, maxiter::Int)
    op = _build_pardiso_linear_operator(H, S, sigma)
    try
        vals, vecs, nconv, niter, nmult, resid = Arpack.eigs(op; nev=nev, which=:LM,
            tol=tol, maxiter=maxiter, ncv=ncv)
        if nconv == 0
            return vals, vecs, nconv, niter, nmult, resid
        end
        λs = sigma .+ inv.(vals[1:nconv])
        vecs = vecs[:, 1:nconv]
        return λs, vecs, nconv, niter, nmult, resid
    finally
        _pardiso_cleanup!(op)
    end
end

function _arpack_eigs_native(H::Hermitian{ComplexF64,Matrix{ComplexF64}},
    S::Union{Nothing,Hermitian{ComplexF64,Matrix{ComplexF64}}}, nev::Int,
    sigma::Float64, ncv::Union{Nothing,Int}, tol::Float64, maxiter::Int)
    kwargs = (; nev=nev, sigma=sigma, which=:LM, tol=tol, maxiter=maxiter)
    if ncv !== nothing
        kwargs = merge(kwargs, (; ncv=ncv))
    end
    if S === nothing
        return Arpack.eigs(H; kwargs...)
    else
        return Arpack.eigs(H, S; kwargs...)
    end
end

function _arpack_eigs(H::AbstractMatrix{ComplexF64}, S::Union{Nothing,AbstractMatrix{ComplexF64}},
    nev::Int, sigma::Float64, ncv::Union{Nothing,Int}, tol::Float64, maxiter::Int,
    solver::Symbol, pardiso_min_size::Int, fallback_full::Bool)
    size(H, 1) == size(H, 2) || error("H must be square.")
    solver_choice = solver
    if solver_choice === :auto
        solver_choice = (_pardiso_active() && sigma !== nothing && size(H, 1) >= pardiso_min_size) ? :pardiso : :native
    end
    if solver_choice === :pardiso
        @debug "Using Pardiso shift-invert solver" size=size(H, 1)
        try
            ncv_eff = ncv === nothing ? min(size(H, 1), max(2 * nev + 2, 20)) : min(size(H, 1), Int(ncv))
            return _arpack_eigs_pardiso(H, S, nev, sigma, ncv_eff, tol, maxiter)
        catch err
            if !fallback_full
                rethrow(err)
            end
            @warn "Pardiso-based shift-invert failed; falling back to native solver." exception=(err, catch_backtrace())
            solver_choice = :native
        end
    end
    # native ARPACK path needs DENSE Hermitian; this is a memory cliff for large sparse models.
    ncv_native = ncv === nothing ? nothing : min(size(H, 1), Int(ncv))
    Hherm = Hermitian(Matrix(H))
    Sherm = S === nothing ? nothing : Hermitian(Matrix(S))
    return _arpack_eigs_native(Hherm, Sherm, nev, sigma, ncv_native, tol, maxiter)
end

@memoize k function eigs_near(tm::AbstractTBModel, k::AbstractVector{<:Real},
    reference::Real; nev::Integer=8, window::Union{Nothing,Real}=nothing,
    ncv::Union{Nothing,Integer}=nothing, sigma::Union{Nothing,Real}=nothing,
    tol::Real=sqrt(eps(Float64)), maxiter::Integer=800,
    compute_residuals::Bool=false, dense_cutoff::Integer=DEFAULT_DENSE_THRESHOLD,
    fallback_full::Bool=true, solver::Symbol=DEFAULT_SOLVER,
    pardiso_min_size::Integer=DEFAULT_PARDISO_MIN_SIZE)::PartialHermEig
    n = tm.norbits
    neff = clamp(Int(nev), 1, n)
    densethresh = clamp(Int(dense_cutoff), 1, n)
    ref = Float64(reference)
    if fallback_full && (n <= densethresh || neff >= n)
        return _dense_eigs(tm, k, ref, window)
    end
    # Bypass `getH`/`getS` (which force dense conversion) and call the order-(0,0,0)
    # Bloch sums directly. For SparseTBModel these return SparseMatrixCSC, avoiding
    # an O(n²) dense allocation that would blow memory at >50k orbits.
    Hmat = HopTB.getdH(tm, (0, 0, 0), k)
    Smat = tm.isorthogonal ? nothing : HopTB.getdS(tm, (0, 0, 0), k)
    sigma_val = sigma === nothing ? ref : Float64(sigma)
    vals, vecs, nconv, niter, nmult, resid = _arpack_eigs(Hmat, Smat, neff, sigma_val,
        ncv, Float64(tol), Int(maxiter), solver, Int(pardiso_min_size), fallback_full)
    if nconv == 0
        if fallback_full
            return _dense_eigs(tm, k, ref, window)
        else
            error("ARPACK failed to converge for k-point; no eigenpairs obtained.")
        end
    end
    energies = Float64.(real(vals[1:nconv]))
    vecs = vecs[:, 1:nconv]
    ill_threshold = _ill_threshold()
    energies, vecs = _orthonormalize_eigenpairs(Hmat, Smat, energies, vecs, ill_threshold)
    order = sortperm(abs.(energies .- ref))
    energies = energies[order]
    vecs = vecs[:, order]
    compute_res = compute_residuals ? _compute_residuals(Hmat, Smat, energies, vecs) :
        fill(NaN, length(energies))
    if window !== nothing
        win = Float64(window)
        mask = abs.(energies .- ref) .<= win
        energies = energies[mask]
        vecs = vecs[:, mask]
        compute_res = compute_res[mask]
    end
    return PartialHermEig(energies, vecs, compute_res, ref)
end

# ---- Block shift-invert Lanczos (Stage 4b) ---------------------------------
#
# Rationale (measured, 110k-orbital slab): a Pardiso triangular solve costs
# 3.17 s for one RHS but 0.074 s/RHS for 128 at once (43x amortization), and
# ARPACK converged nev=500 with a Krylov space of exactly ncv~2*nev in a
# single restart. So: build the same-size Krylov basis with the SAME
# factorization but in block solves, then one Rayleigh-Ritz.

_smul(S::Nothing, X::AbstractMatrix) = copy(X)
_smul(S::SparseMatrixCSC, X::AbstractMatrix) = HopTB.spmm_threaded(S, X)
_smul(S::AbstractMatrix, X::AbstractMatrix) = S * X

# Y = A^{-1} B via Pardiso phase 33, chunked so each call has <= chunk RHS.
function _pardiso_block_solve!(Y::Matrix{ComplexF64}, op::PardisoShiftOp,
        B::AbstractMatrix{ComplexF64}, chunk::Int)
    n, m = size(B)
    for lo in 1:chunk:m
        hi = min(lo + chunk - 1, m)
        t0 = time_ns()
        Pardiso.set_phase!(op.solver, Pardiso.SOLVE_ITERATIVE_REFINE)
        Pardiso.pardiso(op.solver, view(Y, :, lo:hi), op.matrix, view(B, :, lo:hi))
        MATVEC_STATS[:n] += hi - lo + 1
        MATVEC_STATS[:solve] += (time_ns() - t0) / 1e9
    end
    return Y
end

# S-metric whitening of W (returns W, SW with W' S W = I); drops near-null
# directions. SW must equal S*W on entry and is kept consistent.
function _s_whiten!(W::Matrix{ComplexF64}, SW::Matrix{ComplexF64}; rank_tol::Float64=1e-10)
    G = Hermitian(W' * SW)
    F = eigen(G)
    keep = F.values .> rank_tol * max(maximum(abs.(F.values)), eps())
    any(keep) || return W[:, 1:0], SW[:, 1:0]
    T = F.vectors[:, keep] * Diagonal(1 ./ sqrt.(F.values[keep]))
    return W * T, SW * T
end

@doc raw"""
```julia
eigs_near_block(tm, k, reference; nstates=500, block=128, basis_mult=2.2,
    tol=1e-7, max_basis_mult=3.2, window=nothing)::PartialHermEig
```

Block shift-invert Lanczos for the `nstates` generalized eigenpairs of
(H(k), S(k)) nearest `reference`. Factorizes H-σS once (σ=`reference`), builds
an S-orthonormal Krylov basis of ≈`basis_mult*nstates` columns using
`block`-wide multi-RHS Pardiso solves, then Rayleigh–Ritz. If the wanted
residuals exceed `tol`, the basis is extended up to `max_basis_mult*nstates`.
Returns the same `PartialHermEig` as `eigs_near` (residuals always computed).

!!! warning "Measured no-go for clustered shift-invert spectra (July 2026)"
    At 110k orbitals / nev=500 this runs 5-10x faster than ARPACK but leaves
    10-30% of wanted states unconverged even at 4.2x basis: block width does
    not substitute for Krylov depth in a clustered transformed spectrum, and
    ARPACK (2 solves/state, one restart) is depth-optimal there. Kept for
    experimentation; production uses `:arpack`. Evidence: jobs 15050414,
    15056526; notesforcode PDF section 9.5.
"""
@memoize k function eigs_near_block(tm::AbstractTBModel, k::AbstractVector{<:Real},
    reference::Real; nstates::Integer=500, block::Integer=128,
    basis_mult::Real=2.2, tol::Real=1e-7, max_basis_mult::Real=3.2,
    window::Union{Nothing,Real}=nothing)::PartialHermEig
    n = tm.norbits
    ref = Float64(reference)
    nst = clamp(Int(nstates), 1, n)
    blk = clamp(Int(block), 1, n)
    Hmat = HopTB.getdH(tm, (0, 0, 0), k)
    Smat = tm.isorthogonal ? nothing : HopTB.getdS(tm, (0, 0, 0), k)
    op = _build_pardiso_linear_operator(Hmat, Smat, ref)
    try
        target = clamp(ceil(Int, Float64(basis_mult) * nst), blk, n)
        maxdim = clamp(ceil(Int, Float64(max_basis_mult) * nst), target, n)
        rng = MersenneTwister(0x5eed1 + Int(nst) + 7919 * round(Int, 1e6 * sum(abs.(k))))

        V  = Matrix{ComplexF64}(undef, n, 0)   # S-orthonormal basis
        SV = Matrix{ComplexF64}(undef, n, 0)   # S * V (cached: projections need no SpMM)
        # Start in the range of the operator: one op application to the random
        # block suppresses ghost Ritz pairs from raw noise directions.
        W0 = randn(rng, ComplexF64, n, blk)
        W  = _pardiso_block_solve!(Matrix{ComplexF64}(undef, n, blk), op, _smul(Smat, W0), blk)

        θ = Float64[]; C = Matrix{ComplexF64}(undef, 0, 0)
        HV = Matrix{ComplexF64}(undef, n, 0)
        while true
            # Pre-projection S-norms: any column whose S-norm collapses during
            # projection is numerically inside span(V); whitening it back to
            # unit norm would amplify rounding noise into a fake basis vector
            # that poisons Rayleigh-Ritz. Deflate such columns instead.
            SW0 = _smul(Smat, W)
            pre = sqrt.(abs.(vec(sum(conj.(W) .* SW0; dims=1))))
            # orthogonalize W against V in the S-metric (CGS x2, cached SV)
            if size(V, 2) > 0
                for _ in 1:2
                    W .-= V * (SV' * W)
                end
            end
            SW = _smul(Smat, W)
            post = sqrt.(abs.(vec(sum(conj.(W) .* SW; dims=1))))
            keep = post .> 1e-8 .* max.(pre, eps())
            if get(ENV, "HOPTB_BLOCK_DEBUG", "") == "true"
                shr = post ./ max.(pre, eps())
                @info "block-append" basis=size(V, 2) newcols=size(W, 2) min_shrink=minimum(shr) max_shrink=maximum(shr) ndrop=count(!, keep)
            end
            if !all(keep)
                W = W[:, keep]; SW = SW[:, keep]
            end
            if size(W, 2) > 0
                W, SW = _s_whiten!(W, SW; rank_tol=1e-8)
            end
            exhausted = size(W, 2) == 0
            if exhausted && size(V, 2) == 0
                @warn "eigs_near_block: start block deflated to nothing."
                break
            end
            V  = hcat(V, W)
            SV = hcat(SV, SW)
            basis_done = exhausted || size(V, 2) >= target
            # next Lanczos block: W <- A^{-1} (S * last block)
            if !basis_done
                W = _pardiso_block_solve!(Matrix{ComplexF64}(undef, n, size(SW, 2)), op, SW, blk)
                continue
            end
            # Rayleigh-Ritz on span(V): Ssub = I by construction.
            # Selection must be residual-aware: an unconverged Krylov direction
            # can place a ghost Ritz value arbitrarily close to ref. Accept the
            # nstates nearest-ref pairs only when every one of them is converged;
            # otherwise grow the basis (ghosts either converge or drift away).
            HV = _smul(Hmat, V)
            F = eigen(Hermitian(V' * HV))
            order = sortperm(abs.(F.values .- ref))
            ncand = min(length(order), nst)
            sel = order[1:ncand]
            θ = F.values[sel]
            C = F.vectors[:, sel]
            Rmat = HV * C - (SV * C) * Diagonal(θ)
            res = vec(sqrt.(sum(abs2, Rmat; dims=1)))
            if maximum(res) <= Float64(tol) || size(V, 2) >= maxdim || exhausted
                maximum(res) > Float64(tol) &&
                    @warn "eigs_near_block: max wanted residual $(maximum(res)) > tol at basis $(size(V,2))/$maxdim (exhausted=$exhausted)."
                break
            end
            # Extend the basis with inverse-iteration images of the WORST
            # Ritz vectors (Davidson-style targeted expansion) instead of
            # blindly continuing the Krylov chain: S*X_sel = SV*C is a GEMM on
            # the cached SV, and each A^{-1} application contracts precisely
            # the states that failed the residual test (window-edge states).
            worst = sortperm(res; rev=true)
            nexp = min(length(worst), 2 * blk)
            Wrhs = SV * C[:, worst[1:nexp]]
            target = min(maxdim, max(target + nexp, ceil(Int, 1.15 * target)))
            W = _pardiso_block_solve!(Matrix{ComplexF64}(undef, n, nexp), op, Wrhs, blk)
        end
        if isempty(θ) && size(V, 2) > 0
            HV = _smul(Hmat, V)
            F = eigen(Hermitian(V' * HV))
            order = sortperm(abs.(F.values .- ref))
            sel = order[1:min(nst, length(order))]
            θ = F.values[sel]
            C = F.vectors[:, sel]
        end
        isempty(θ) && error("eigs_near_block: no Ritz pairs obtained.")
        X = V * C
        Rmat = HV * C - (SV * C) * Diagonal(θ)
        res = vec(sqrt.(sum(abs2, Rmat; dims=1)))
        energies = Float64.(θ)
        if window !== nothing
            win = Float64(window)
            mask = abs.(energies .- ref) .<= win
            energies = energies[mask]; X = X[:, mask]; res = res[mask]
        end
        return PartialHermEig(energies, X, res, ref)
    finally
        _pardiso_cleanup!(op)
    end
end

function eigs_window(tm::AbstractTBModel, k::AbstractVector{<:Real}, reference::Real,
    window::Real; nev::Integer=8, min_states::Integer=2, max_tries::Integer=6,
    growth_factor::Real=1.5, kwargs...)::PartialHermEig
    n = tm.norbits
    target = clamp(Int(min_states), 1, n)
    current_nev = clamp(Int(nev), 1, n)
    tries = 0
    last_spec = nothing
    while tries < max_tries
        spec = eigs_near(tm, k, reference; nev=current_nev, window=window, kwargs...)
        if length(spec) >= target || current_nev >= n
            return spec
        end
        tries += 1
        last_spec = spec
        current_nev = min(n, max(current_nev + 1, Int(cld(current_nev * growth_factor, 1))))
        if current_nev >= n
            break
        end
    end
    if last_spec === nothing
        return eigs_near(tm, k, reference; nev=current_nev, window=window, kwargs...)
    end
    return last_spec
end

end
const DEFAULT_ILL_THRESHOLD = 5e-4

_ill_threshold() = begin
    val = get(ENV, "HOPTB_ILL_THRESHOLD", nothing)
    if val === nothing || isempty(strip(val))
        return DEFAULT_ILL_THRESHOLD
    end
    try
        return parse(Float64, strip(val))
    catch
        return DEFAULT_ILL_THRESHOLD
    end
end
