module HopTB

# memoization module
include("memoize.jl")
# mesh module
include("mesh.jl")
# utilities module
include("utilities.jl")
# parallel module
include("parallel.jl")


# direct members
include("model.jl")
include("sparse_model.jl")
using .SparseModel: SparseTBModel
include("basic.jl")
include("sparse_extensions.jl")
include("fs.jl")


# submodules
include("sparse_calc.jl")
include("optics.jl")
include("bs.jl")
include("hall.jl")
include("conduct.jl")
include("nonlinear_transport.jl")
include("group.jl")
include("wannier.jl")
include("topology.jl")
include("floquet.jl")
include("structure.jl")
include("magnetism.jl")


# indepedent modules
include("response.jl")

# interface module
include("interface.jl")

# example module
include("zoo.jl")
end
