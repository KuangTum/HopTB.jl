# print_like_openmx.jl — 以 OpenMX analysis 风格打印 S 与 r
using Printf, LinearAlgebra

# ---------- 读 4 元组打包的 3D 向量 ----------
function read_packed_f64(io, num)::Matrix{Float64}
    buf = Vector{Float64}(undef, 4*num); read!(io, buf)
    M = reshape(buf, 4, num)
    Matrix(M[2:4, :])
end
function read_packed_i32(io, num)::Matrix{Int}
    buf = Vector{Int32}(undef, 4*num); read!(io, buf)
    M = reshape(buf, 4, num)
    Int.(M[2:4, :])
end
# 直接按列主序读一个块矩阵（rows=邻居轨道, cols=中心轨道）
function read_block_f64(io, rows::Int, cols::Int)
    M = Matrix{Float64}(undef, rows, cols); read!(io, M); M
end

# ---------- 读取 overlap-only .scfout ----------
function read_olpr(filepath::String)
    open(filepath, "r") do io
        hdr = Vector{Int32}(undef, 7); read!(io, hdr)
        atomnum      = Int(hdr[1])
        spinP_switch = Int(hdr[2]) & 0x03
        Catomnum     = Int(hdr[3]); Latomnum = Int(hdr[4]); Ratomnum = Int(hdr[5])
        TCpyCell     = Int(hdr[6])
        order_max    = Int(hdr[7])

        atv     = read_packed_f64(io, TCpyCell+1)     # 3 x (TCpyCell+1)
        atv_ijk = read_packed_i32(io, TCpyCell+1)     # 3 x (TCpyCell+1)

        TNO  = Int.(read!(io, Vector{Int32}(undef, atomnum)))
        FNAN = Int.(read!(io, Vector{Int32}(undef, atomnum)))

        natn = [Int.(read!(io, Vector{Int32}(undef, FNAN[i]+1))) for i in 1:atomnum]
        ncn  = [Int.(read!(io, Vector{Int32}(undef, FNAN[i]+1))) for i in 1:atomnum]

        tv  = read_packed_f64(io, 3)
        rtv = read_packed_f64(io, 3)
        gbuf = Vector{Float64}(undef, 4*atomnum); read!(io, gbuf)
        G = reshape(gbuf, 4, atomnum); Gxyz = Matrix(G[2:4, :])

        OLP = [Vector{Matrix{Float64}}(undef, FNAN[i]+1) for i in 1:atomnum]
        for i in 1:atomnum, h in 1:(FNAN[i]+1)
            B = natn[i][h]
            OLP[i][h] = read_block_f64(io, TNO[B], TNO[i])
        end
        OLP_r = ntuple(_->([Vector{Matrix{Float64}}(undef, FNAN[i]+1) for i in 1:atomnum]), 3)
        for α in 1:3, i in 1:atomnum, h in 1:(FNAN[i]+1)
            B = natn[i][h]
            M = read_block_f64(io, TNO[i], TNO[B])  # centre×neighbour
            OLP_r[α][i][h] = M'                     # transpose to neighbour×centre
        end

        return (; atomnum, spinP_switch, Catomnum, Latomnum, Ratomnum,
                 TCpyCell, order_max, atv, atv_ijk, TNO, FNAN, natn, ncn,
                 tv, rtv, Gxyz, OLP, OLP_r)
    end
end

# ---------- 打印工具（行内浮点） ----------
@inline function print_row(vals::AbstractVector{<:Real})
    for k in 1:length(vals)
        @printf(" % .7f", vals[k])
    end
    print('\n')
end

# 取 Rn 的“整数索引”和“(i,j,k) 三元组”
function rn_index_and_triplet(atv_ijk::Matrix{Int}, c::Int)
    ncols = size(atv_ijk, 2)
    # 自块通常用 -1 或 0 标；规整化索引与三元组
    # 在 .scfout 中 ncn 是 0-based；索引越界返回零向量
    if c < 0 || c + 1 > ncols
        return (c, (0,0,0))
    else
        cc = c + 1                    # 转为 Julia 的 1-based 列号
        return (c, (atv_ijk[1,cc], atv_ijk[2,cc], atv_ijk[3,cc]))
    end
end

# ---------- 以 OpenMX analysis 风格打印 ----------
function print_like_openmx(data; printS=false, printRx=false, printRy=false, printRz=false)
    atomnum = data.atomnum
    TNO     = data.TNO
    FNAN    = data.FNAN
    natn    = data.natn
    ncn     = data.ncn
    atv_ijk = data.atv_ijk
    OLP     = data.OLP
    OLP_r   = data.OLP_r

    # S
    if printS
        println("Overlap matrix")
        for i in 1:atomnum
            for h in 1:(FNAN[i]+1)
                B = natn[i][h]
                rn_idx, (ri,rj,rk) = rn_index_and_triplet(atv_ijk, ncn[i][h])
                # local index 从 0 开始更贴近 OpenMX 输出
                @printf("global index=%d  local index=%d (global=%d, Rn=%d)\n",
                        i, h-1, B, rn_idx)
                M = OLP[i][h]
                for r in 1:size(M,1); print_row(M[r, :]); end
            end
        end
    end

    # r_x / r_y / r_z
    function print_r(dir::Int, title::String)
        println(title)
        for i in 1:atomnum
            for h in 1:(FNAN[i]+1)
                B = natn[i][h]
                rn_idx, (ri,rj,rk) = rn_index_and_triplet(atv_ijk, ncn[i][h])
                # 有些 OpenMX 变体会把 Rn 打成四元（0 i j k），这里两种都给：先三元，再整数
                @printf("global index=%d  local index=%d (global=%d, Rn=%d %d %d %d)\n",
                        i, h-1, B, rn_idx, ri, rj, rk)
                M = OLP_r[dir][i][h]
                for r in 1:size(M,1); print_row(M[r, :]); end
            end
        end
    end
    if printRx; print_r(1, "Overlap matrix with position operator x"); end
    if printRy; print_r(2, "Overlap matrix with position operator y"); end
    if printRz; print_r(3, "Overlap matrix with position operator z"); end
end

# ---------- 主程序 ----------
function main()
    if length(ARGS) < 1
        println("Usage: julia -q print_like_openmx.jl /path/to/openmx_olpr.scfout [--S] [--rx] [--ry] [--rz]")
        return
    end
    filepath = ARGS[1]
    flags = Set(Symbol.(strip.(replace.(ARGS[2:end], "--"=>""))))
    printS  = (:S in flags)  || isempty(flags)
    printRx = (:rx in flags) || isempty(flags)
    printRy = (:ry in flags) || isempty(flags)
    printRz = (:rz in flags) || isempty(flags)

    data = read_olpr(filepath)
    print_like_openmx(data; printS=printS, printRx=printRx, printRy=printRy, printRz=printRz)
end
main()
