module InverseHyperplanes

using Symbolics: Symbolics, @variables
using ParametricMCPs: ParametricMCPs, ParametricMCP
using BlockArrays: BlockArray, Block, mortar, blocks
using LinearAlgebra: norm_sqr
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    PolygonEnvironment,
    ProductDynamics,
    TimeSeparableTrajectoryGameCost,
    TrajectoryGame,
    GeneralSumCostStructure,
    num_players,
    time_invariant_linear_dynamics,
    unstack_trajectory,
    stack_trajectories,
    state_dim,
    control_dim,
    state_bounds,
    control_bounds,
    OpenLoopStrategy,
    JointStrategy,
    RecedingHorizonStrategy,
    rollout, 
    horizon
using LazySets: LazySets
using BlockArrays: mortar, blocks, BlockArray, Block, blocksizes
using PATHSolver: PATHSolver
using LinearAlgebra: norm_sqr, norm
using ProgressMeter: ProgressMeter
using Plots
using Zygote
using Printf 
using UnPack: @unpack
using Random: MersenneTwister
using DataFrames
using CSV 
using Statistics: median
using Optimisers

include("parametric_optimization_problem.jl")
export ParametricOptimizationProblem, solve, total_dim

include("parametric_game.jl")
export ParametricGame

include("dynamics.jl")
export cwh_satellite_2D

include("trajectory_game_utils.jl")
export setup_trajectory_game, extract_states

include("visualization.jl")
export visualize_rotating_hyperplanes, build_parametric_game

include("forward_game.jl")
export forward

include("inverse_game.jl")
export inverse

end # module
