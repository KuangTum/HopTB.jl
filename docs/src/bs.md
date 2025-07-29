# HopTB.BandStructure

## Functions
```@docs
HopTB.BandStructure.getbs
HopTB.BandStructure.getdos
HopTB.BandStructure.getjdos
HopTB.BandStructure.clteig
HopTB.BandStructure.get_fermi_surfaces
```

## Parallel DOS calculation

`getdos` distributes the $k$-point mesh across all active Julia workers using the `Distributed` module. Launch Julia with multiple processes (e.g. `julia -p 4`) or add workers via `Distributed.addprocs` before calling `getdos`.

The following script `examples/mydos_script.jl` computes the DOS of the sample boron nitride model in parallel and saves it to `dos.dat`:

```julia
using Distributed
if nworkers() == 1
    addprocs()
end

using HopTB

# model and parameters
tm = HopTB.Zoo.getBN()
ωs = -5:0.01:5
nkmesh = [50, 50, 1]

dos = HopTB.BandStructure.getdos(tm, ωs, nkmesh; ϵ=0.1)

open("dos.dat", "w") do io
    for (ω, d) in zip(ωs, dos)
        println(io, "$(ω) $(d)")
    end
end
```

To run this script on a Slurm cluster you can use an `sbatch` file similar to

```bash
#!/bin/bash
#SBATCH --job-name=dos
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00

module load julia
julia -p $SLURM_NTASKS examples/mydos_script.jl
```

`getdos` will automatically use all `$SLURM_NTASKS` processes.
