module Utilities

using LinearAlgebra, Distributed, HCubature, Spglib


"""
```julia
integrate(f::Function; rtol::Float64=√eps(),
    atol::Float64=0.0, maxevals::Int64=typemax(Int64))::Tuple
```

Integrate `f` from -∞ to ∞. The second argument of the returned tuple is error.
"""
function integrate(f::Function; rtol::Float64=√eps(),
    atol::Float64=0.0, maxevals::Int64=typemax(Int64))
    return hquadrature(-1, 1, rtol=rtol, atol=atol, maxevals=maxevals) do t
        f(t/(1-t^2))*(1+t^2)/((1-t^2)^2)
    end
end


"""
```julia
integrate(f::Function, x0::Vector{Float64}, x1::Vector{Float64};
    constant_components::Vector{Int64}=zeros(Int64, 0), rtol::Float64=√eps(),
    atol::Float64=0.0, maxevals::Int64=typemax(Int64), initdiv::Int64=1)
```

Integrate function `f` from `x0` to `x1` by HCubature package.

`f` is a function that takes a `Vector{Float64}` and yields a number or a matrix.
The error of integartion is estimated by `norm(..., Inf)`.
`atol` and `rtol` are absolute tolerance and respective tolerance respectively.
The integration will stop whenever `atol` or `rtol` is reached or when
`maxevals` evaluations of `f` have been done. Initially, the volume is divided
into `initdiv` segments along each dimension.

It is common that `f` is independent on some components of its argument `x`,
in this case, one can specify these components by `constant_components` and
the integration will generally be boosted.
"""
function integrate(f::Function, x0::Vector{Float64}, x1::Vector{Float64};
    constant_components::Vector{Int64}=zeros(Int64, 0), rtol::Float64=√eps(),
    atol::Float64=0.0, maxevals::Int64=typemax(Int64), initdiv::Int64=1)
    xlen = length(x0)
    xlen > 0 || error("length of x0 should be larger than 0.")
    xlen == length(x1) || error("length of x0 should be the same as x1.")

    varying_components = deleteat!(collect(1:xlen), constant_components)

    nf = let x0=x0, f=f
        function _nf(nx)
            x = x0[:]
            x[varying_components] = nx
            return f(x)
        end
    end

    constant_volume = prod(x1[constant_components]-x0[constant_components])

    return hcubature(nf, x0[varying_components], x1[varying_components],
        atol=atol/constant_volume, rtol=rtol, maxevals=maxevals,
        norm=x->norm(x, Inf), initdiv=initdiv).*constant_volume
end


function _dividevolume(x0::Vector{Float64}, x1::Vector{Float64}, ndiv::Int64)
    @assert length(x0) > 0
    @assert length(x1) == length(x0)
    dxend = (x1[end]-x0[end])/ndiv
    x0send = collect(0:ndiv-1)*dxend.+x0[end]
    x1send = collect(1:ndiv)*dxend.+x0[end]
    if length(x0) ==  1
        return (reshape(x0send, (1, length(x0send))), reshape(x1send, (1, length(x1send))))
    end
    lastx0s, lastx1s = _dividevolume(x0[1:end-1], x1[1:end-1], ndiv)
    lastsize = size(lastx0s, 2)
    x0s = zeros(length(x0), lastsize*ndiv)
    x1s = zeros(length(x1), lastsize*ndiv)

    for cnt in 1:ndiv
        x0s[:, lastsize*(cnt-1)+1:lastsize*cnt] = [lastx0s; ones(1, lastsize)*x0send[cnt]]
        x1s[:, lastsize*(cnt-1)+1:lastsize*cnt] = [lastx1s; ones(1, lastsize)*x1send[cnt]]
    end
    return (x0s, x1s)
end


"""
```julia
pintegrate(f::Function, x0::Vector{Float64}, x1::Vector{Float64};
    ndiv::Int64=1, constant_components::Vector{Int64}=zeros(Int64, 0),
    rtol::Float64=√eps(), atol::Float64=0.0, maxevals::Int64=typemax(Int64),
    initdiv::Int64=1)
```

Parallel integration of `f`. Most parameters have the same meaning as
`integrate`. And the integration region is divided into `ndiv` segments
in every dimension. The divided regions will be integrated on different processes.
"""
function pintegrate(f::Function, x0::Vector{Float64}, x1::Vector{Float64};
    ndiv::Int64=1, constant_components::Vector{Int64}=zeros(Int64, 0),
    rtol::Float64=√eps(), atol::Float64=0.0, maxevals::Int64=typemax(Int64),
    initdiv::Int64=1)

    xlen = length(x0)
    varying_components = deleteat!(collect(1:xlen), constant_components)
    x0svarying, x1svarying = _dividevolume(x0[varying_components], x1[varying_components], ndiv)
    x0s = zeros(xlen, size(x0svarying, 2))
    x0s[varying_components, :] = x0svarying
    x0s[constant_components, :] .= x0[constant_components]
    x1s = zeros(xlen, size(x1svarying, 2))
    x1s[varying_components, :] = x1svarying
    x1s[constant_components, :] .= x1[constant_components]
    njobs = size(x0s, 2)
    results = Vector{Any}(undef, njobs)
    jobidx = 1
    getjob() = (jobidx+=1; jobidx-1)
    @sync begin
        for proc in workers()
            @async begin
                while true
                    myjobidx = getjob()
                    if myjobidx > njobs
                        break
                    end
                    results[myjobidx] = fetch(@spawnat proc integrate(
                        f, x0s[:, myjobidx], x1s[:, myjobidx],
                        constant_components=constant_components,
                        rtol=rtol, atol=atol/njobs, maxevals=maxevals,
                        initdiv=initdiv))
                end
            end
        end
    end
    return (sum([results[cnt][1] for cnt in 1:njobs]), sum([results[cnt][2] for cnt in 1:njobs]))
end


function splitkpts(kpts::AbstractMatrix{Float64}, nsplit::Int64)
    nkpts = size(kpts, 2)
    kptslist = Vector{Matrix{Float64}}()
    snkpts = floor(Int64, nkpts / nsplit)
    kptsstart = 1
    for ikpts in 1:nsplit
        if ikpts <= (nkpts - snkpts * nsplit)
            append!(kptslist, [kpts[:, kptsstart:(kptsstart + snkpts)]])
            kptsstart += snkpts + 1
        else
            append!(kptslist, [kpts[:, kptsstart:(kptsstart + snkpts - 1)]])
            kptsstart += snkpts
        end
    end
    return kptslist
end


function constructmeshkpts(nkmesh::Vector{Int64}; offset::Vector{Float64}=[0.0, 0.0, 0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    length(nkmesh) == 3 || throw(ArgumentError("nkmesh in wrong size."))
    nkpts = prod(nkmesh)
    kpts = zeros(3, nkpts)
    ik = 1
    for ikx in 1:nkmesh[1], iky in 1:nkmesh[2], ikz in 1:nkmesh[3]
        kpts[:, ik] = [
            (ikx-1)/nkmesh[1]*(k2[1]-k1[1])+k1[1],
            (iky-1)/nkmesh[2]*(k2[2]-k1[2])+k1[2],
            (ikz-1)/nkmesh[3]*(k2[3]-k1[3])+k1[3]
        ]
        ik += 1
    end
    return kpts.+offset
end


"""
    construct_irreducible_kmesh(
        lattice::AbstractMatrix{<:Real},
        frac_positions::AbstractMatrix{<:Real},
        atom_types::AbstractVector{<:Integer},
        nkmesh::AbstractVector{<:Integer};
        is_shift::NTuple{3,Bool}=(false, false, false),
        symprec::Float64=1e-5,
        spinful::Bool=false,
    )

Generate irreducible Monkhorst-Pack k-points and symmetry weights using Spglib.

`lattice` columns are direct lattice vectors in angstrom, `frac_positions` are fractional
coordinates (3 × Natoms), and `atom_types` lists atomic numbers. `nkmesh` gives the full
grid size. `is_shift` controls half-grid shifts and `spinful` disables time-reversal.

Returns `(kpts, weights)` where `kpts` is a `3 × N_ir` matrix in reduced coordinates and
`weights` sums to one.
"""
function construct_irreducible_kmesh(
    lattice::AbstractMatrix{<:Real},
    frac_positions::AbstractMatrix{<:Real},
    atom_types::AbstractVector{<:Integer},
    nkmesh::AbstractVector{<:Integer};
    is_shift::NTuple{3,Bool}=(false, false, false),
    symprec::Float64=1e-5,
    spinful::Bool=false,
)
    size(lattice) == (3, 3) || throw(ArgumentError("lattice must be 3×3."))
    size(frac_positions, 1) == 3 || throw(ArgumentError("frac_positions must have 3 rows."))
    length(atom_types) == size(frac_positions, 2) ||
        throw(ArgumentError("atom_types length must match number of positions."))
    length(nkmesh) == 3 || throw(ArgumentError("nkmesh must have length 3."))

    mesh_vec = collect(Int.(nkmesh))
    cell = Spglib.SpglibCell(lattice, frac_positions, Vector{Int}(atom_types))
    bz = Spglib.get_ir_reciprocal_mesh(
        cell,
        mesh_vec,
        symprec;
        is_shift=collect(is_shift),
        is_time_reversal=!spinful,
    )

    ir_indices = unique(bz.ir_mapping_table)
    lookup = Dict(id => idx for (idx, id) in enumerate(ir_indices))
    counts = zeros(Int, length(ir_indices))
    for idx in bz.ir_mapping_table
        counts[lookup[idx]] += 1
    end

    mesh_vec = Float64.(bz.mesh)
    shift_vec = 0.5 .* Float64.(bz.is_shift)
    kpts = Matrix{Float64}(undef, 3, length(ir_indices))
    for (col, idx) in enumerate(ir_indices)
        address = bz.grid_address[idx]
        kpts[:, col] = (Float64.(address) .+ shift_vec) ./ mesh_vec
    end

    weights = counts ./ sum(counts)
    return kpts, weights
end


function _hasfield(tm, name::Symbol)
    return hasproperty(tm, name)
end

function _getfield_or_missing(tm, name::Symbol)
    _hasfield(tm, name) ? getfield(tm, name) : missing
end

function _get_spinful(tm)
    val = _getfield_or_missing(tm, :isspinful)
    return ismissing(val) ? false : Bool(val)
end

function _get_site_positions(tm)
    site_pos = _getfield_or_missing(tm, :site_positions)
    return ismissing(site_pos) ? missing : Float64.(site_pos)
end

function _orbital_positions(tm)
    positions_dict = _getfield_or_missing(tm, :positions)
    ismissing(positions_dict) && error("orbital position matrices are not available for this model.")
    R0_key = nothing
    for R in keys(positions_dict)
        if all(R .== 0)
            R0_key = R
            break
        end
    end
    R0_key === nothing && error("model does not contain on-site position data (R = 0).")
    blocks = positions_dict[R0_key]
    norb = size(blocks[1], 1)
    cart = Matrix{Float64}(undef, 3, norb)
    for α in 1:3
        block = blocks[α]
        for i in 1:norb
            cart[α, i] = real(block[i, i])
        end
    end
    return cart
end

function _infer_site_types(tm, nsites::Int)
    site_norbits = _getfield_or_missing(tm, :site_norbits)
    orbital_types = _getfield_or_missing(tm, :orbital_types)
    if ismissing(site_norbits) && ismissing(orbital_types)
        return ones(Int, nsites)
    end
    actual_sites = if !ismissing(site_norbits)
        length(site_norbits)
    elseif !ismissing(orbital_types)
        length(orbital_types)
    else
        nsites
    end
    signatures = Vector{Int}(undef, actual_sites)
    cache = Dict{Any,Int}()
    for i in 1:actual_sites
        parts = Any[]
        if !ismissing(site_norbits)
            push!(parts, site_norbits[i])
        end
        if !ismissing(orbital_types)
            push!(parts, Tuple(orbital_types[i]))
        end
        sig = isempty(parts) ? i : parts
        signatures[i] = get!(cache, sig, length(cache) + 1)
    end
    return signatures
end

function _infer_orbital_types(tm, norb::Int)
    site_norbits = _getfield_or_missing(tm, :site_norbits)
    if ismissing(site_norbits)
        return ones(Int, norb)
    end
    site_types = _infer_site_types(tm, length(site_norbits))
    expanded = Int[]
    for (stype, count) in zip(site_types, site_norbits)
        append!(expanded, fill(stype, count))
    end
    return length(expanded) == norb ? expanded : ones(Int, norb)
end

function _cluster_positions(cart::AbstractMatrix{<:Real}, tol::Float64)
    tol > 0 || error("cluster tolerance must be positive.")
    scale = 1 / tol
    unique = Vector{Vector{Float64}}()
    keys = Dict{NTuple{3,Int64},Int}()
    for j in 1:size(cart, 2)
        key = ntuple(i -> round(Int64, cart[i, j] * scale), 3)
        if !haskey(keys, key)
            push!(unique, Float64.(cart[:, j]))
            keys[key] = length(unique)
        end
    end
    return isempty(unique) ? zeros(Float64, 3, 0) : hcat(unique...)
end

function _choose_positions(tm, mode::Symbol; cluster_tol::Float64=1e-2)
    selected = mode
    site_pos = _get_site_positions(tm)
    if selected === :auto
        selected = ismissing(site_pos) ? :orbital : :site
    end
    if selected === :site
        !ismissing(site_pos) || error("site positions are missing; specify positions=:orbital.")
        return site_pos, :site
    elseif selected === :orbital
        cart = _orbital_positions(tm)
        clustered = _cluster_positions(cart, cluster_tol)
        return clustered, :orbital
    else
        error("positions must be :auto, :site, or :orbital")
    end
end

function _default_atom_types(tm, count::Int, which::Symbol)
    if which === :site
        types = _infer_site_types(tm, count)
        return length(types) == count ? types : ones(Int, count)
    else
        types = _infer_orbital_types(tm, count)
        return length(types) == count ? types : ones(Int, count)
    end
end

"""
    construct_irreducible_kmesh(
        tm,
        nkmesh::AbstractVector{<:Integer};
        atom_types::Union{Nothing,AbstractVector{<:Integer}}=nothing,
        positions::Symbol=:auto,
        is_shift::NTuple{3,Bool}=(false, false, false),
        symprec::Float64=1e-5,
        spinful::Union{Nothing,Bool}=nothing,
    )

Convenience wrapper that infers lattice, atomic positions, and spin setting from a `TBModel`.

`positions` chooses whether to use stored site positions (`:site`) or orbital centers (`:orbital`).
The default `:auto` prefers site positions when available. If `atom_types` is omitted the helper
derives species labels from site metadata when possible; otherwise all atoms are treated as the
same species.
"""
function construct_irreducible_kmesh(
    tm,
    nkmesh::AbstractVector{<:Integer};
    atom_types::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    positions::Symbol=:auto,
    cluster_tol::Float64=1e-2,
    is_shift::NTuple{3,Bool}=(false, false, false),
    symprec::Float64=1e-5,
    spinful::Union{Nothing,Bool}=nothing,
)
    lattice = Matrix(tm.lat)
    cart_positions, mode = _choose_positions(tm, positions; cluster_tol=cluster_tol)
    frac_positions = lattice \ cart_positions
    species = atom_types === nothing ? _default_atom_types(tm, size(frac_positions, 2), mode) : Vector{Int}(atom_types)
    length(species) == size(frac_positions, 2) ||
        throw(ArgumentError("length of atom_types must equal number of positions."))
    spinflag = isnothing(spinful) ? _get_spinful(tm) : spinful
    return construct_irreducible_kmesh(
        lattice,
        frac_positions,
        species,
        nkmesh;
        is_shift=is_shift,
        symprec=symprec,
        spinful=spinflag,
    )
end


@doc raw"""
```julia
getmesh(
    gridsize::Vector{Int64};
    offset::Vector{Float64}=[0.0, 0.0, 0.0],
    v1::Vector{Float64}=[0.0, 0.0, 0.0],
    v2::Vector{Float64}=[1.0, 1.0, 1.0]
) --> Vector{Vector{Float64}}
```

Construct a unform grid.
"""
function getmesh(
    meshsize::Vector{Int64};
    offset::Vector{Float64}=[0.0, 0.0, 0.0],
    v1::Vector{Float64}=[0.0, 0.0, 0.0],
    v2::Vector{Float64}=[1.0, 1.0, 1.0]
)
    length(meshsize) == 3 || throw(ArgumentError("meshsize in wrong size."))
    npoints = prod(meshsize)
    mesh = Vector{Vector{Float64}}()
    nx, ny, nz = meshsize
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        push!(mesh, [
            (ix - 1) / nx * (v2[1] - v1[1]) + v1[1] + offset[1],
            (iy - 1) / ny * (v2[2] - v1[2]) + v1[2] + offset[2],
            (iz - 1) / nz * (v2[3] - v1[3]) + v1[3] + offset[3],
        ])
    end
    return mesh
end


"""
k in kpath and returned kpts are in reduced coordinates.
"""
function constructlinekpts(kpath::AbstractMatrix{Float64}, rlat::AbstractMatrix{Float64},
    pnkpts::Int64; connect_end_points::Bool=false)
    size(kpath, 1) == 3 || error("size(kpath, 1) should be 3.")
    (connect_end_points == false && isodd(size(kpath, 2))) &&
        error("size(kpath, 1) should be even if connect_end_points is false.")

    if connect_end_points
        nkpath = zeros(3, 2*size(kpath, 2)-2)
        for i in 1:size(kpath, 2)-1
            nkpath[:, 2*i-1] = kpath[:, i]
            nkpath[:, 2*i] = kpath[:, i+1]
        end
    else
        nkpath = kpath
    end
    npath = size(nkpath, 2)÷2
    kpts = zeros(3, npath*pnkpts)
    kdist = zeros(npath*pnkpts)

    for ipath in 1:npath
        kstart = nkpath[:, 2*ipath-1]
        kend = nkpath[:, 2 * ipath]
        dk = (kend-kstart)/(pnkpts-1)
        if ipath == 1
            kdiststart = 0.0
        else
            kdiststart = kdist[(ipath-1)*pnkpts]
        end
        for ikpt in 1:pnkpts
            kpts[:, ikpt+(ipath-1)*pnkpts] = kstart+(ikpt-1)*dk
            kdist[ikpt+(ipath-1)*pnkpts] = norm((ikpt-1)*rlat*dk)+kdiststart
        end
    end

    return (kdist, kpts)
end


function constructlinekpts(kpath::AbstractMatrix{<:Real}, pnkpts::Int64; connect_end_points::Bool=false)
    size(kpath, 1) == 3 || error("size(kpath, 1) should be 3.")
    (connect_end_points == false && isodd(size(kpath, 2))) &&
        error("size(kpath, 1) should be even if connect_end_points is false.")

    if connect_end_points
        nkpath = zeros(3, 2*size(kpath, 2)-2)
        for i in 1:size(kpath, 2)-1
            nkpath[:, 2*i-1] = kpath[:, i]
            nkpath[:, 2*i] = kpath[:, i+1]
        end
    else
        nkpath = kpath
    end
    npath = size(nkpath, 2)÷2
    kpts = zeros(3, npath*pnkpts)

    for ipath in 1:npath
        kstart = nkpath[:, 2*ipath-1]
        kend = nkpath[:, 2 * ipath]
        dk = (kend-kstart)/(pnkpts-1)
        for ikpt in 1:pnkpts
            kpts[:, ikpt+(ipath-1)*pnkpts] = kstart+(ikpt-1)*dk
        end
    end

    return kpts
end


"""
```julia
fermidirac(T::Float64, E::Float64)
```

Fermi-Dirac distribution. The temperature T is in Kelvin.
E is energy - chemical potential and is in eV.
"""
function fermidirac(T::Float64, E::Float64)
    beta = 1/(8.617333262145e-5 * T)  #1/kBT
    if T == 0.0
        return (sign(-E) + 1.0) / 2 # theta(-E) function -E>0,1
    else
        return 1.0 / (exp(E * beta) + 1.0)
    end
end

#fermidirac(T, ϵn-μ) to get Fermi-Dirac distribution

"""
    dfermi_dE(T, E)

∂f/∂E of Fermi–Dirac distribution.
* `T` : temperature in K
* `E` : energy (already E - μ) in eV
Returns value in 1/eV.
"""
function dfermi_dE(T::Float64, E::Float64)
    T == 0.0 && return 0.0
    f  = fermidirac(T, E)
    β  = 1/(8.617333262145e-5 * T)
    return -β * f * (1 - f)  #no NaN
end


function safe_fetch(f::Future)
    r = fetch(f)
    if r isa RemoteException
        @error ("Exception " * string(r.captured.ex) * " found in worker " * string(r.pid)  *
            "\n" * join(map((x)->string(x[1]) * " at worker " * string(x[2]),
                r.captured.processed_bt),
            "\n"))
        return nothing
    end
    return r
end

function eye(::Type{T}, n) where T
    return Matrix{T}(I, n, n)
end

eye(n) = eye(Float64, n)

function distance_on_circle(a::Float64, b::Float64)
    a′ = mod(a, 1.0)
    b′ = mod(b, 1.0)
    return minimum(abs.([a′-b′, a′-b′+1.0, a′-b′-1.0]))
end


function heigen(H)
    return eigen(Hermitian(H))
end


function heigen(H, S)
    return eigen(Hermitian(H), Hermitian(S))
end


function undefs(::Type{T}, dims...) where T
    return Array{T}(undef, dims...)
end


function undefs(dims...)
    return undefs(Float64, dims...)
end

function splitvector(v::Vector{T}, N::Int64)::Vector{Vector{T}} where T
    l = length(v)
    n = floor(Int64, l/N)
    r = Vector{Vector{T}}()
    cnt = 1
    for i in 1:(n+1)*N-l
        push!(r, v[cnt:cnt+n-1])
        cnt += n
    end
    for i in 1:l-n*N
        push!(r, v[cnt:cnt+n])
        cnt += n+1
    end
    @assert cnt == l+1
    return r
end

end
