module Conduct

using LinearAlgebra, Distributed
import Spglib
using ..HopTB
using ..HopTB.Utilities:
    fermidirac,
    dfermi_dE,
    constructmeshkpts,
    splitkpts,
    construct_irreducible_kmesh,
    _choose_positions,
    _default_atom_types,
    _get_spinful
using ..HopTB.Parallel: ParallelFunction, claim!, stop!, parallel_sum

export getahc


function _split_kmesh_with_weights(kpts::AbstractMatrix{Float64}, weights::AbstractVector{Float64}, nsplit::Int)
    nkpts = size(kpts, 2)
    nsplit = max(1, min(nsplit, nkpts))
    counts = fill(div(nkpts, nsplit), nsplit)
    remainder = rem(nkpts, nsplit)
    for i in 1:remainder
        counts[i] += 1
    end
    chunks = Vector{Tuple{Matrix{Float64},Vector{Float64}}}(undef, nsplit)
    start_idx = 1
    for i in 1:nsplit
        stop_idx = start_idx + counts[i] - 1
        chunks[i] = (Matrix(kpts[:, start_idx:stop_idx]), collect(weights[start_idx:stop_idx]))
        start_idx = stop_idx + 1
    end
    return chunks
end


function _getahc(atm::AbstractTBModel, α::Int64, β::Int64, kpts::AbstractMatrix{Float64};
    Ts::Vector{Float64} = [0.0], μs::Vector{Float64} = [0.0])
    nkpts = size(kpts, 2)
    itgrd = zeros(ComplexF64, length(Ts), length(μs))
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals, egvecs = geteig(atm, k)
        order = [0, 0, 0]
        order[α] = 1
        Sbar_α = egvecs' * getdS(atm, Tuple(order), k) * egvecs
        order = [0, 0, 0]
        order[β] = 1
        Sbar_β = egvecs' * getdS(atm, Tuple(order), k) * egvecs
        Abar_α = egvecs' * HopTB.getAw(atm, α, k) * egvecs
        Abar_β = egvecs' * HopTB.getAw(atm, β, k) * egvecs
        Dα = HopTB.getD(atm, α, k)
        Dβ = HopTB.getD(atm, β, k)
        order = [0, 0, 0]
        order[α] = 1
        dAw_βα = HopTB.getdAw(atm, β, Tuple(order), k)
        order = [0, 0, 0]
        order[β] = 1
        dAw_αβ = HopTB.getdAw(atm, α, Tuple(order), k)
        Ωbar_αβ = egvecs' * (dAw_βα - dAw_αβ) * egvecs
        tmp1 = Sbar_α*Abar_β
        tmp2 = Sbar_β*Abar_α
        for iT in 1:length(Ts)
            for iμ in 1:length(μs)
                for n in 1:atm.norbits
                    f = fermidirac(Ts[iT], egvals[n]-μs[iμ])
                    itgrd[iT, iμ] +=  f*(Ωbar_αβ[n, n]-tmp1[n, n]+tmp2[n, n])
                end
                for n in 1:atm.norbits, m in 1:atm.norbits
                    fm = fermidirac(Ts[iT], egvals[m] - μs[iμ])
                    fn = fermidirac(Ts[iT], egvals[n] - μs[iμ])
                    itgrd[iT, iμ] += (fm - fn) * (im * Dα[n, m] * Dβ[m, n] + Dα[n, m] * Abar_β[m, n] - Dβ[n, m] * Abar_α[m, n])
                end
            end
        end
    end
    return real.(itgrd)
end


@doc raw"""
```julia
getahc(atm::AbstractTBModel, α::Int64, β::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0])::Matrix{Float64}
```

Calculate anomalous Hall conductivity $σ^{αβ}$.

Anomalous Hall conductivity is defined by
```math
σ^{αβ}=-\frac{e^2}{ħ}\int\frac{d\boldsymbol{k}}{(2pi)^3}f_nΩ_{nn}^{αβ}.
```

The returned matrix $σ^{αβ}[m, n]$ is AHC for temperature Ts[m] and
chemical potential μs[n].

The returned AHC is in unit (Ω⋅cm)^-1.
"""
function getahc(atm::AbstractTBModel, α::Int64, β::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64} = [0.0], μs::Vector{Float64} = [0.0])
    @assert size(nkmesh, 1) == 3
    nkpts = prod(nkmesh)
    kpts = HopTB.Utilities.constructmeshkpts(nkmesh)
    kptslist = HopTB.Utilities.splitkpts(kpts, nworkers())

    jobs = Vector{Future}()
    for iw in 1:nworkers()
        job = @spawn _getahc(atm, α, β, kptslist[iw]; Ts = Ts, μs = μs)
        append!(jobs, [job])
    end


    σs = zeros(Float64, length(Ts), length(μs))
    for iw in 1:nworkers()
        σs += HopTB.Utilities.safe_fetch(jobs[iw])
    end

    bzvol = abs(dot(cross(atm.rlat[:, 1], atm.rlat[:, 2]), atm.rlat[:, 3]))
    return σs * bzvol / nkpts * (-98.130728142) # -e**2/(hbar*(2pi)^3)*1.0e10/100
end


function _collect_berry_curvature(atm::AbstractTBModel, α::Int64, β::Int64, kpts::AbstractMatrix{Float64})
    nkpts = size(kpts, 2)
    berry_curvature = zeros(atm.norbits, nkpts)
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals, egvecs = geteig(atm, k)
        order = [0, 0, 0]; order[α] = 1; Sbar_α = egvecs' * getdS(atm, Tuple(order), k) * egvecs
        order = [0, 0, 0]; order[β] = 1; Sbar_β = egvecs' * getdS(atm, Tuple(order), k) * egvecs
        Abar_α = egvecs' * HopTB.getAw(atm, α, k) * egvecs
        Abar_β = egvecs' * HopTB.getAw(atm, β, k) * egvecs
        Dα = HopTB.getD(atm, α, k)
        Dβ = HopTB.getD(atm, β, k)
        order = [0, 0, 0]; order[α] = 1; dAw_βα = HopTB.getdAw(atm, β, Tuple(order), k)
        order = [0, 0, 0]; order[β] = 1; dAw_αβ = HopTB.getdAw(atm, α, Tuple(order), k)
        Ωbar_αβ = egvecs' * (dAw_βα - dAw_αβ) * egvecs
        berry_curvature[:, ik] = real.(diag(Ωbar_αβ - Sbar_α * Abar_β + Sbar_β * Abar_α - im * Dα * Dβ + 
            im * Dβ * Dα - Dα * Abar_β + Abar_β * Dα + Dβ * Abar_α - Abar_α * Dβ))
    end
    return berry_curvature
end

"""
```julia
collect_berry_curvature(atm::AbstractTBModel, α::Int64, β::Int64, kpts::AbstractMatrix{Float64})::Matrix{Float64}
```

Collect berry curvature.

Standard units is used (eV and Å).

The returned matrix Ω[n, ik] is berry curvature for band n at ik point.
"""
function collect_berry_curvature(atm::AbstractTBModel, α::Int64, β::Int64, kpts::AbstractMatrix{Float64})
    nkpts = size(kpts, 2)

    kptslist = HopTB.Utilities.splitkpts(kpts, nworkers())
    jobs = Vector{Future}()
    for iw in 1:nworkers()
        job = @spawn _collect_berry_curvature(atm, α, β, kptslist[iw])
        append!(jobs, [job])
    end

    result = zeros((atm.norbits, 0))
    for iw in 1:nworkers()
        result = cat(result, HopTB.Utilities.safe_fetch(jobs[iw]), dims = (2,))
    end
    return result
end


function shc_worker(kpts::Matrix{Float64}, tm::TBModel, α::Int64, β::Int64, γ::Int64,
    Ts::Vector{Float64}, μs::Vector{Float64}, ϵ::Float64)
    result = zeros(length(Ts), length(μs))
    nkpts = size(kpts, 2)
    for ik in 1:nkpts
        k = kpts[:, ik]
        vα = getvelocity(tm, α, k); vβ = getvelocity(tm, β, k)
        sγ = getspin(tm, γ, k)
        jαγ = (sγ*vα+vα*sγ)/2
        egvals, _ = geteig(tm, k)
        for (iT, T) in enumerate(Ts), (iμ, μ) in enumerate(μs)
            for n in 1:tm.norbits
                ϵn = egvals[n]
                fn = fermidirac(T, ϵn-μ)
                for m in 1:tm.norbits
                    ϵm = egvals[m]
                    result[iT, iμ] += -fn*imag(jαγ[n, m]*vβ[m, n]/((ϵn-ϵm)^2+ϵ^2))
                end
            end
        end
    end
    return result
end

function ahc_worker(job::Tuple{Matrix{Float64},Vector{Float64}}, tm::AbstractTBModel, 
    ωs::Vector{Float64}, T::Float64, μ::Float64, δ::Float64)
    kpts, weights = job
    result = zeros(Float64, length(ωs))
    for (ik, w) in enumerate(weights)
        k = kpts[:, ik]
        vα = getvelocity(tm, 1, k); vβ = getvelocity(tm, 2, k)
        egvals, _ = geteig(tm, k)
        for (iω, ω)  in enumerate(ωs)
            for n in 1:tm.norbits
                ϵn = egvals[n]
                fn = fermidirac(T, ϵn-μ)
                for m in [1:n-1; n+1:tm.norbits]  # Combine ranges, skipping n
                    ϵm = egvals[m]
                    fm = fermidirac(T, ϵm-μ)
                    fnm = (fn-fm)
                    result[iω] += w * fnm * imag((vα[n, m]*vβ[m, n]) / ((ϵm - ϵn)^2 - (ω + im*δ)^2)) 
                end    
            end
        end
    end
    return result
end


function ahcdc_worker(job::Tuple{Matrix{Float64},Vector{Float64}}, tm::AbstractTBModel, 
    T::Float64, μ::Float64)
    kpts, weights = job
    result = 0.0
    for (ik, w) in enumerate(weights)
        k = kpts[:, ik]
        vα = getvelocity(tm, 1, k); vβ = getvelocity(tm, 2, k)
        egvals, _ = geteig(tm, k)
        for n in 1:tm.norbits
            ϵn = egvals[n]
            fn = fermidirac(T, ϵn-μ)
            for m in [1:n-1; n+1:tm.norbits]  # Combine ranges, skipping n
                ϵm = egvals[m]
                result += w * fn * 2*imag((vα[n, m]*vβ[m, n]) / ((ϵm - ϵn)^2))
            end   
        end
    end
    return result
end

const VELOCITY_DEGENERACY_EPS = 1e-7

function _mesh_kwargs_dict(mesh_kwargs)
    return isempty(mesh_kwargs) ? Dict{Symbol,Any}() : Dict{Symbol,Any}(mesh_kwargs)
end

function _extract_mesh_kwargs(mesh_kwargs)
    kw = _mesh_kwargs_dict(mesh_kwargs)
    allow_fractional = pop!(kw, :allow_fractional_rotations, false)
    return kw, allow_fractional
end

function _collect_point_group_rotations(tm::AbstractTBModel; mesh_kwargs...)
    kw = _mesh_kwargs_dict(mesh_kwargs)
    lattice = Matrix{Float64}(tm.lat)
    positions_mode = get(kw, :positions, :auto)
    cluster_tol = get(kw, :cluster_tol, 1e-2)
    cart_positions, pos_mode = _choose_positions(tm, positions_mode; cluster_tol=cluster_tol)
    frac_positions = lattice \ cart_positions
    atom_types_kw = get(kw, :atom_types, nothing)
    species = atom_types_kw === nothing ? _default_atom_types(tm, size(frac_positions, 2), pos_mode) : Vector{Int}(atom_types_kw)
    length(species) == size(frac_positions, 2) ||
        throw(ArgumentError("length of atom_types must equal number of positions."))
    spinful_kw = get(kw, :spinful, nothing)
    spinflag = isnothing(spinful_kw) ? _get_spinful(tm) : spinful_kw
    symprec = get(kw, :symprec, 1e-5)
    cell = Spglib.SpglibCell(lattice, frac_positions, Vector{Int}(species))
    rotations, _ = try
        Spglib.get_symmetry(cell, symprec; is_time_reversal=!spinflag)
    catch err
        if err isa MethodError
            Spglib.get_symmetry(cell, symprec)
        else
            rethrow()
        end
    end
    unique_rotations = unique(rotations)
    lattice_inv = inv(lattice)
    rot_cart = Matrix{Float64}[]
    for R in unique_rotations
        Rmat = Matrix{Float64}(R)
        push!(rot_cart, lattice * Rmat * lattice_inv)
    end
    isempty(rot_cart) && push!(rot_cart, Matrix{Float64}(I, 3, 3))
    return rot_cart
end

function _deduplicate_rotations(rotations::Vector{Matrix{Float64}}; tol::Float64=1e-8)
    unique_rots = Matrix{Float64}[]
    for R in rotations
        if !any(norm(R .- existing) < tol for existing in unique_rots)
            push!(unique_rots, R)
        end
    end
    isempty(unique_rots) && push!(unique_rots, Matrix{Float64}(I, 3, 3))
    return unique_rots
end

function _convert_rotations_to_kspace(rotations::Vector{Matrix{Float64}}, rlat::AbstractMatrix{<:Real})
    B = Matrix{Float64}(rlat)
    Binv = inv(B)
    converted = Matrix{Float64}[]
    for R in rotations
        push!(converted, Binv * R * B)
    end
    return converted
end

function _filter_inplane_rotations(rotations::Vector{Matrix{Float64}}; tol::Float64=1e-6, allow_fractional::Bool=false)
    filtered = Matrix{Float64}[]
    for R in rotations
        if abs(R[3, 3] - 1.0) < tol && abs(R[1, 3]) < tol && abs(R[2, 3]) < tol &&
           abs(R[3, 1]) < tol && abs(R[3, 2]) < tol && abs(det(R) - 1.0) < tol
            # Only enforce proper in-plane rotation in Cartesian here.
            # Integer-like filtering should be done in k-space basis.
            push!(filtered, R)
        end
    end
    return _deduplicate_rotations(filtered; tol=tol)
end

function _filter_integer_like_kspace(rotations_k::Vector{Matrix{Float64}}; tol::Float64=1e-6, allow_fractional::Bool=false)
    allow_fractional && return _deduplicate_rotations(rotations_k; tol=tol)
    filtered = Matrix{Float64}[]
    for Rk in rotations_k
        block = @view Rk[1:2, 1:2]
        is_integer_like = true
        for i in 1:2, j in 1:2
            if abs(block[i, j] - round(block[i, j])) > tol
                is_integer_like = false
                break
            end
        end
        is_integer_like && push!(filtered, Rk)
    end
    return _deduplicate_rotations(filtered; tol=tol)
end

function _kspace_to_cart(rot_k::Vector{Matrix{Float64}}, rlat::AbstractMatrix{<:Real})
    B = Matrix{Float64}(rlat)
    Binv = inv(B)
    out = Matrix{Float64}[]
    for Rk in rot_k
        push!(out, B * Rk * Binv)
    end
    return out
end

function _rotation_maps_mesh!(Rk::AbstractMatrix{<:Real}, kpts::AbstractMatrix{<:Real}; tol::Float64=1e-6)
    # Check if Rk maps every k in the mesh to another k in the same mesh (mod 1)
    nk = size(kpts, 2)
    # Build canonical list once
    canon = Matrix{Float64}(undef, 3, nk)
    for j in 1:nk
        canon[:, j] = _canonicalize_kpt(view(kpts, :, j); tol=tol)
    end
    for j in 1:nk
        kr = _canonicalize_kpt(Rk * view(kpts, :, j); tol=tol)
        found = false
        @inbounds for i in 1:nk
            if maximum(abs.(kr .- view(canon, :, i))) < tol
                found = true
                break
            end
        end
        found || return false
    end
    return true
end

function _filter_mesh_compatible_rotations(kpts_full::AbstractMatrix{<:Real}, rotations_k::Vector{Matrix{Float64}}; tol::Float64=1e-6)
    keep = Matrix{Float64}[]
    for Rk in rotations_k
        _rotation_maps_mesh!(Rk, kpts_full; tol=tol) && push!(keep, Rk)
    end
    return _deduplicate_rotations(keep; tol=tol)
end

function _canonicalize_kpt(k::AbstractVector{<:Real}; tol::Float64=1e-8)
    canon = Vector{Float64}(undef, length(k))
    for i in eachindex(k)
        val = Float64(k[i])
        val -= floor(val)
        if abs(val - 1.0) < tol || abs(val) < tol
            val = 0.0
        end
        canon[i] = val
    end
    return canon
end

function _orbit_from_rep(rep::Vector{Float64}, rotations::Vector{Matrix{Float64}}; tol::Float64=1e-6)
    orbit = Vector{Vector{Float64}}()
    for R in rotations
        rotated = _canonicalize_kpt(R * rep; tol=tol)
        if !any(maximum(abs.(rotated .- existing)) < tol for existing in orbit)
            push!(orbit, rotated)
        end
    end
    return orbit
end

function _reduce_kmesh_with_rotations(kpts::Matrix{Float64}, rotations::Vector{Matrix{Float64}}; tol::Float64=1e-6)
    nk = size(kpts, 2)
    canon_pts = Matrix{Float64}(undef, size(kpts, 1), nk)
    for j in 1:nk
        canon_pts[:, j] = _canonicalize_kpt(view(kpts, :, j); tol=tol)
    end
    I3 = Matrix{Float64}(I, 3, 3)
    if isempty(rotations)
        rotations = [I3]
    elseif !any(norm(R .- I3) < tol for R in rotations)
        push!(rotations, I3)
    end
    visited = falses(nk)
    reps = Matrix{Float64}(undef, size(kpts, 1), 0)
    weights = Float64[]
    for i in 1:nk
        if visited[i]
            continue
        end
        rep = canon_pts[:, i]
        orbit = _orbit_from_rep(rep, rotations; tol=tol)
        count = 0
        for j in i:nk
            if visited[j]
                continue
            end
            kj = canon_pts[:, j]
            if any(maximum(abs.(kj .- member)) < tol for member in orbit)
                visited[j] = true
                count += 1
            end
        end
        reps = hcat(reps, rep)
        push!(weights, count)
    end
    total = sum(weights)
    weights .= weights ./ total
    return reps, weights
end

function _symmetrize_conductivity_tensor(σ::Array{ComplexF64,3}, rotations::Vector{Matrix{Float64}})
    isempty(rotations) && return σ
    σ_sym = zeros(ComplexF64, size(σ))
    tmp_left = Matrix{ComplexF64}(undef, 3, 3)
    tmp = Matrix{ComplexF64}(undef, 3, 3)
    for R in rotations
        Rtr = transpose(R)
        for iω in axes(σ, 3)
            σ_slice = @view σ[:, :, iω]
            mul!(tmp_left, R, σ_slice)
            mul!(tmp, tmp_left, Rtr)
            @views σ_sym[:, :, iω] .+= tmp
        end
    end
    σ_sym ./= length(rotations)
    return σ_sym
end

function _symmetrize_conductivity_tensor_2d(σ::Array{ComplexF64,3}, rotations::Vector{Matrix{Float64}})
    isempty(rotations) && return σ
    σ_sym = zeros(ComplexF64, size(σ))
    tmp_left = Matrix{ComplexF64}(undef, 2, 2)
    tmp = Matrix{ComplexF64}(undef, 2, 2)
    for R in rotations
        R2 = @view R[1:2, 1:2]
        R2tr = transpose(R2)
        for iω in axes(σ, 3)
            σ_slice = @view σ[:, :, iω]
            mul!(tmp_left, R2, σ_slice)
            mul!(tmp, tmp_left, R2tr)
            @views σ_sym[:, :, iω] .+= tmp
        end
    end
    σ_sym ./= length(rotations)
    return σ_sym
end

function AC_tensor_worker(job::Tuple{Matrix{Float64},Vector{Float64}}, tm::AbstractTBModel,
    ωs::Vector{Float64}, T::Float64, μ::Float64, δ::Float64)
    kpts, weights = job
    nω = length(ωs)
    result = zeros(ComplexF64, 3, 3, nω)
    velocity_products = Matrix{ComplexF64}(undef, 3, 3)
    prefactors = Vector{ComplexF64}(undef, nω)
    imδ = im * δ
    for (ik, w) in enumerate(weights)
        k = kpts[:, ik]
        velocities = ntuple(α -> getvelocity(tm, α, k), 3)
        egvals, _ = geteig(tm, k)
        for n in 1:tm.norbits
            ϵn = egvals[n]
            fn = fermidirac(T, ϵn - μ)
            for m in 1:tm.norbits
                ϵm = egvals[m]
                Δ = ϵn - ϵm
                for α in 1:3, β in 1:3
                    velocity_products[α, β] = velocities[α][n, m] * velocities[β][m, n]
                end
                absΔ = abs(Δ)
                if absΔ > VELOCITY_DEGENERACY_EPS
                    fm = fermidirac(T, ϵm - μ)
                    coeff = w * (fn - fm) / Δ
                    @inbounds for (iω, ω) in pairs(ωs)
                        prefactors[iω] = coeff / (Δ + ω + imδ)
                    end
                else
                    coeff = w * dfermi_dE(T, ϵn - μ)
                    @inbounds for (iω, ω) in pairs(ωs)
                        prefactors[iω] = coeff / (ω + imδ)
                    end
                end
                @inbounds for (iω, pref) in pairs(prefactors)
                    @views result[:, :, iω] .+= pref .* velocity_products
                end
            end
        end
    end
    return result
end

function AC_tensor_worker_2d(job::Tuple{Matrix{Float64},Vector{Float64}}, tm::AbstractTBModel,
    ωs::Vector{Float64}, T::Float64, μ::Float64, δ::Float64)
    kpts, weights = job
    nω = length(ωs)
    result = zeros(ComplexF64, 2, 2, nω)
    velocity_products = Matrix{ComplexF64}(undef, 2, 2)
    prefactors = Vector{ComplexF64}(undef, nω)
    imδ = im * δ
    for (ik, w) in enumerate(weights)
        k = kpts[:, ik]
        v1 = getvelocity(tm, 1, k)
        v2 = getvelocity(tm, 2, k)
        egvals, _ = geteig(tm, k)
        for n in 1:tm.norbits
            ϵn = egvals[n]
            fn = fermidirac(T, ϵn - μ)
            for m in 1:tm.norbits
                ϵm = egvals[m]
                Δ = ϵn - ϵm
                velocity_products[1,1] = v1[n,m]*v1[m,n]
                velocity_products[1,2] = v1[n,m]*v2[m,n]
                velocity_products[2,1] = v2[n,m]*v1[m,n]
                velocity_products[2,2] = v2[n,m]*v2[m,n]
                absΔ = abs(Δ)
                if absΔ > VELOCITY_DEGENERACY_EPS
                    fm = fermidirac(T, ϵm - μ)
                    coeff = w * (fn - fm) / Δ
                    @inbounds for (iω, ω) in pairs(ωs)
                        prefactors[iω] = coeff / (Δ + ω + imδ)
                    end
                else
                    coeff = w * dfermi_dE(T, ϵn - μ)
                    @inbounds for (iω, ω) in pairs(ωs)
                        prefactors[iω] = coeff / (ω + imδ)
                    end
                end
                @inbounds for (iω, pref) in pairs(prefactors)
                    @views result[:, :, iω] .+= pref .* velocity_products
                end
            end
        end
    end
    return result
end

@doc raw"""
```julia
getAC(tm::TBModel, α::Int64, β::Int64, ωs::Vector{Float64}, nkmesh::Vector{Int64};
    T::Float64, μ::Float64;gs::Int64=1, δ::Float64=0.05)::Vector{ComplexF64}
'''note
The 2D unit is e**2/hbar, while the 3D unit is (Ω ⋅ cm)^-1.
kubo-greenwood Formula from PHYSICAL REVIEW B 98, 115115 (2018)

'''    

Conductivity from this routine is symmetrized over the crystal point group when
`use_symmetry=true` (default).

"""

function _build_weighted_mesh(tm::AbstractTBModel, nkmesh::Vector{Int64}; use_symmetry::Bool=false, mesh_kwargs...)
    kw, allow_fractional = _extract_mesh_kwargs(mesh_kwargs)
    if use_symmetry
        if nkmesh[3] == 1
            kpts_full = constructmeshkpts(nkmesh)
            # 2D: keep proper in-plane rotations in Cartesian, then test integer-likeness in k-space
            rotations_cart = _collect_point_group_rotations(tm; kw...)
            rotations_cart = _filter_inplane_rotations(rotations_cart; allow_fractional=true)
            rotations_k = _convert_rotations_to_kspace(rotations_cart, tm.rlat)
            rotations_k = _filter_integer_like_kspace(rotations_k; allow_fractional=allow_fractional)
            rotations_k = _filter_mesh_compatible_rotations(kpts_full, rotations_k)
            kpts, weights = _reduce_kmesh_with_rotations(kpts_full, rotations_k)
            println("2D IBZ kpts: \n", kpts)
            println("2D IBZ weights: ", weights)
            return kpts, weights
        else
            kpts, weights = construct_irreducible_kmesh(tm, nkmesh; kw...)
            return kpts, weights
        end
    else
        kpts = constructmeshkpts(nkmesh)
        nkpts = size(kpts, 2)
        weights = fill(1.0 / nkpts, nkpts)
        return kpts, weights
    end
end

function getAC(tm::AbstractTBModel, α::Int64, β::Int64, ωs::Vector{Float64}, nkmesh::Vector{Int64},
    T::Float64, μ::Float64; gs::Int64=1, δ::Float64=0.05, use_symmetry::Bool=true,
    symmetrize::Bool=use_symmetry, symmetrize_match_ibz::Bool=true, mesh_kwargs...)
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    # 2D slab optimization: only in-plane components (xx, xy, yx, yy) are relevant.
    if nkmesh[3] == 1 && (α == 3 || β == 3)
        return zeros(ComplexF64, length(ωs))
    end

    kw, allow_fractional = _extract_mesh_kwargs(mesh_kwargs)
    kpts, weights = _build_weighted_mesh(tm, nkmesh; use_symmetry=use_symmetry, mesh_kwargs...)
    
    # Save k-points and weights together
    open("kpts_and_weights.txt", "w") do f
        println(f, "K-points and their corresponding weights:")
        println(f, "Total k-points: $(size(kpts, 2))")
        println(f, "K-point dimensions: $(size(kpts, 1))")
        println(f, "Format: kx  ky  kz  weight")
        println(f, "=" ^ 50)
        for i in 1:size(kpts, 2)
            if size(kpts, 1) == 3
                println(f, "$(kpts[1,i])  $(kpts[2,i])  $(kpts[3,i])  $(weights[i])")
            elseif size(kpts, 1) == 2
                println(f, "$(kpts[1,i])  $(kpts[2,i])  0.0  $(weights[i])")
            else
                kpt_str = join([string(kpts[j,i]) for j in 1:size(kpts,1)], "  ")
                println(f, "$kpt_str  $(weights[i])")
            end
        end
        println(f, "=" ^ 50)
        println(f, "Sum of weights: $(sum(weights))")
    end
    
    chunks = _split_kmesh_with_weights(kpts, weights, nworkers())

    if nkmesh[3] == 1
        pf = ParallelFunction(AC_tensor_worker_2d, tm, ωs, T, μ, δ)
        println("nworkers: ", nworkers(), ", jobs: ", length(chunks))
        for chunk in chunks
            pf(chunk)
        end
        σ2 = zeros(ComplexF64, 2, 2, length(ωs))
        for _ in 1:length(chunks)
            σ2 .+= claim!(pf)
        end
        stop!(pf)
        if symmetrize
            # Option A (default): use the exact subgroup that reduced the mesh (strict equivalence)
            if symmetrize_match_ibz
                kpts_full = constructmeshkpts(nkmesh)
                rot_cart_all  = _collect_point_group_rotations(tm; kw...)
                rot_cart_geom = _filter_inplane_rotations(rot_cart_all; allow_fractional=true)
                rot_k_geom    = _convert_rotations_to_kspace(rot_cart_geom, tm.rlat)
                rot_k_used    = _filter_integer_like_kspace(rot_k_geom; allow_fractional=allow_fractional)
                rot_k_used    = _filter_mesh_compatible_rotations(kpts_full, rot_k_used)
                rot_cart_used = _kspace_to_cart(rot_k_used, tm.rlat)
                # Save rot_cart_used to file
                open("rot_cart_used.txt", "w") do f
                    println(f, "rot_cart_used (used for symmetrization): $(length(rot_cart_used)) matrices")
                    for (i, R) in enumerate(rot_cart_used)
                        println(f, "Rotation $i:")
                        for row in eachrow(R)
                            println(f, "  ", join(row, "  "))
                        end
                        println(f)
                    end
                end
                σ2 = _symmetrize_conductivity_tensor_2d(σ2, rot_cart_used)
            else
                # Option B: previous behavior — geometric filter only (may differ from IBZ subgroup)
                rotations_cart = _collect_point_group_rotations(tm; kw...)
                rotations_cart = _filter_inplane_rotations(rotations_cart; allow_fractional=true)
                σ2 = _symmetrize_conductivity_tensor_2d(σ2, rotations_cart)
            end
        end
        bzvol = abs((tm.rlat[:, 1] × tm.rlat[:, 2])⋅tm.rlat[:, 3])
        println("get 2D AC conductivity.")
        σ2 .*= (-im * gs * bzvol / (2 * pi))
        return σ2[α, β, :]
    else
        pf = ParallelFunction(AC_tensor_worker, tm, ωs, T, μ, δ)
        println("nworkers: ", nworkers(), ", jobs: ", length(chunks))
        for chunk in chunks
            pf(chunk)
        end
        σ3 = zeros(ComplexF64, 3, 3, length(ωs))
        for _ in 1:length(chunks)
            σ3 .+= claim!(pf)
        end
        stop!(pf)
        if symmetrize
            rotations = _collect_point_group_rotations(tm; kw...)
            σ3 = _symmetrize_conductivity_tensor(σ3, rotations)
        end
        bzvol = abs((tm.rlat[:, 1] × tm.rlat[:, 2])⋅tm.rlat[:, 3])
        σ3 .*= (-im * gs * bzvol * 98.130728142)
        return σ3[α, β, :]
    end
end


function getAhCω(tm::AbstractTBModel, ωs::Vector{Float64}, nkmesh::Vector{Int64},
    T::Float64, μ::Float64; gs::Int64=1, δ::Float64=0.05, use_symmetry::Bool=false, mesh_kwargs...)
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    kpts, weights = _build_weighted_mesh(tm, nkmesh; use_symmetry=use_symmetry, mesh_kwargs...)
    chunks = _split_kmesh_with_weights(kpts, weights, nworkers())
    pf = ParallelFunction(ahc_worker, tm, ωs, T, μ, δ)
    println("nworkers: ", nworkers(), ", jobs: ", length(chunks))
    for chunk in chunks
        pf(chunk)
    end

    σ_ahc = zeros(Float64, length(ωs))
    for _ in 1:length(chunks)
        σ_ahc += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1] × tm.rlat[:, 2])⋅tm.rlat[:, 3])
    if nkmesh[3] == 1 # 2D case
        println("get 2D AhC conductivity.")
        return gs * σ_ahc * bzvol / (2 * pi) # e**2/hbar
    else
        return gs * σ_ahc * bzvol * 98.130728142 # e**2/(hbar*(2π)^3)*1.0e10/100
    end
end

function getAhCDC(tm::AbstractTBModel, nkmesh::Vector{Int64},
    T::Float64, μ::Float64; gs::Int64=1, use_symmetry::Bool=false, mesh_kwargs...)
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    kpts, weights = _build_weighted_mesh(tm, nkmesh; use_symmetry=use_symmetry, mesh_kwargs...)
    chunks = _split_kmesh_with_weights(kpts, weights, nworkers())
    pf = ParallelFunction(ahcdc_worker, tm, T, μ)
    println("nworkers: ", nworkers(), ", jobs: ", length(chunks))
    for chunk in chunks
        pf(chunk)
    end

    σ_ahc = 0.0
    for _ in 1:length(chunks)
        σ_ahc += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1] × tm.rlat[:, 2])⋅tm.rlat[:, 3])
    if nkmesh[3] == 1 # 2D case
        println("get 2D AhC conductivity.")
        return gs * σ_ahc * bzvol / (2 * pi) # e**2/hbar
    else
        return gs * σ_ahc * bzvol * 98.130728142 # e**2/(hbar*(2π)^3)*1.0e10/100
    end
end


@doc raw"""
```julia
getshc(tm::TBModel, α::Int64, β::Int64, γ::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0], ϵ::Float64=0.1)::Matrix{Float64}
```

Calculate spin Hall conductivity for different temperature (`Ts`, first dimension)
and chemical potential (`μs`, second dimension).

Spin Hall conductivity is defined as
```math
σ_{αβ}^{γ} = eħ\int\frac{d^3 \boldsymbol{k}}{(2π)^3}\sum_n f_n Ω^{γ}_{n,αβ},
```
where the spin Berry curvature is
```math
Ω_{n,αβ}^{γ} = -2 \text{Im} [\sum_{m≠n} \frac{⟨n|\hat{j}_α^γ|m⟩⟨m|\hat{v}_β|n⟩}{(ϵ_n-ϵ_m)^2+ϵ^2}]
```
and the spin current operator is
```math
\hat{j}_α^γ = \frac{1}{2} \{\hat{v}_a, \hat{s}_c\}.
```

Spin Hall conductivity from this function is in ħ/e (Ω*cm)^-1.
"""
function getshc(tm::TBModel, α::Int64, β::Int64, γ::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0], ϵ::Float64=0.1)
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    println("Calculating spin Hall conductivity for $nkpts k-points with mesh $(nkmesh) and $length(Ts) temperatures and $length(μs) chemical potentials.")
    kptslist = splitkpts(constructmeshkpts(nkmesh), nworkers())
    @show size(kptslist)
    @show size(kptslist,1)
    @show size(kptslist[1])
    pf = ParallelFunction(shc_worker, tm, α, β, γ, Ts, μs, ϵ)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs))
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts*98.130728142 # e**2/(hbar*(2π)^3)*1.0e10/100
end


################################################################################
##  Berry curvature dipole
################################################################################


@doc raw"""
```
get_berry_curvature_dipole(
    tm::AbstractTBModel,
    α::Int64,
    β::Int64,
    γ::Int64,
    fss::Vector{FermiSurface}
)::Float64
```

Calculate
```math
\sum_n \int_{\text{FS}_n} \frac{d \sigma}{(2\pi)^3} \Omega_{n}^{\alpha \beta} \frac{v_n^{\gamma}}{|\boldsymbol{v}_n|}
```
which is related to the Berry curvature dipole contribution to the second order photocurrent.
"""
function get_berry_curvature_dipole(
    tm::AbstractTBModel,
    α::Int64,
    β::Int64,
    γ::Int64,
    fss::Vector{FermiSurface}
)
    result = 0.0
    for fs in fss
        result += parallel_sum(ik -> begin
            k = fs.ks[:, ik]
            Ω = get_berry_curvature(tm, α, β, k)[fs.bandidx]
            v = real([getvelocity(tm, i, k)[fs.bandidx, fs.bandidx] for i in 1:3])
            Ω * v[γ] * fs.weights[ik] / norm(v)
        end, 1:size(fs.ks, 2), 0.0)
    end
    return result / (2π)^3
end

end
