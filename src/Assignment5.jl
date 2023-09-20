module Assignment5

using Symbolics: Symbolics, @variables
using ParametricMCPs: ParametricMCPs, ParametricMCP
using BlockArrays: BlockArray, Block, mortar, blocks
using LinearAlgebra: norm_sqr

include("parametric_optimization_problem.jl")
export ParametricOptimizationProblem, solve, total_dim

include("parametric_game.jl")
export ParametricGame

end # module
