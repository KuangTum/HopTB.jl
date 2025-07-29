using Distributed
# Launch additional worker processes if not provided via `-p`.
if nworkers() == 1
    addprocs()
end

using HopTB

# Example tight-binding model (boron nitride)
tm = HopTB.Zoo.getBN()

# Energy grid for DOS
ωs = -5:0.01:5

# Mesh size in k space
nkmesh = [50, 50, 1]

# Compute DOS using all available workers
println("Computing DOS on $(nworkers()) workers ...")
dos = HopTB.BandStructure.getdos(tm, ωs, nkmesh; ϵ=0.1)

# Save results
open("dos.dat", "w") do io
    for (ω, d) in zip(ωs, dos)
        println(io, "$(ω) $(d)")
    end
end
