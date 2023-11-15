""" Utilities for constructing trajectory games, in which each player wishes to # 2B[1]
solve a problem of the form:
                min_{τᵢ}   fᵢ(τ, θ)

where all vehicles must jointly satisfy the constraints
                           g̃(τ, θ) = 0
                           h̃(τ, θ) ≥ 0.

Here, τᵢ is the ith vehicle's trajectory, consisting of states and controls. The shared
constraints g̃ and h̃ incorporate dynamic feasibility, fixed initial condition, actuator and
state limits, environment boundaries, and collision-avoidance.
"""

using Assignment5
using LazySets: LazySets
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
using TrajectoryGamesExamples: planar_double_integrator, animate_sim_steps
using BlockArrays: mortar, blocks, BlockArray, Block, blocksizes
using GLMakie: GLMakie
using Makie: Makie
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

"Utility to set up a trajectory game with rotating hyperplane constraints"
function setup_trajectory_game(; n_players = 2, horizon = ∞, dt = 5.0, n = 0.001, m = 100.0, couples = nothing)

    cost = let
        function stage_cost(x, u, t, θ)
            goals =   θ[Block(2)]
            weights = θ[Block(3)]

            [weights[Block(player)][1] * norm_sqr(x[Block(player)][1:2] - goals[Block(player)]) + weights[Block(player)][2] * norm_sqr(u[Block(player)]) for player in 1:n_players]
        end

        function reducer(stage_costs)
            reduce(.+, stage_costs) ./ length(stage_costs)
        end

        TimeSeparableTrajectoryGameCost(stage_cost, reducer, GeneralSumCostStructure(), 1.0)
    end

    function coupling_constraints(xs, us, θ)
        ωs  = θ[Block(4)]
        α0s = θ[Block(5)]
        ρs  = θ[Block(6)]

        # Hyperplane constraints
        [
            begin
                # Player positions
                x_ego = x[Block(couple[1])]
                x_other = x[Block(couple[2])]

                # Hyperplane normal 
                n_arg = α0s[couple_idx] + ωs[couple_idx] * (ii - 1) * dt
                n = [cos(n_arg), sin(n_arg)]

                # Intersection of hyperplane w/ KoZ
                p = x_other[1:2] + ρs[couple_idx] * n

                # Print value of couple_idx and ii 
                # println("couple_idx = ", couple_idx, " ω = ", ωs[couple_idx], " ρ = ", ρs[couple_idx], ", ii = ", ii)

                # Hyperplane constraint at time ii for couple couple_idx
                n' * (x_ego[1:2] - p)

            end for (couple_idx, couple) in enumerate(couples) for (ii, x) in enumerate(xs)
        ]
    end

    agent_dynamics = cwh_satellite_2D(;
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-1.0, -1.0], ub = [1.0, 1.0]),
        horizon,
        dt,
        n
    )
    dynamics = ProductDynamics([agent_dynamics for _ in 1:n_players])

    TrajectoryGame(dynamics, cost, nothing, coupling_constraints)
end

"Utility for unpacking trajectory."
function unpack_trajectory(flat_trajectory; dynamics::ProductDynamics)
    trajs = Iterators.map(1:num_players(dynamics), blocks(flat_trajectory)) do ii, τ
        horizon = Int(length(τ) / (state_dim(dynamics, ii) + control_dim(dynamics, ii)))
        num_states = state_dim(dynamics, ii) * horizon
        X = reshape(τ[1:num_states], (state_dim(dynamics, ii), horizon))
        U = reshape(τ[(num_states + 1):end], (control_dim(dynamics, ii), horizon))

        (; xs = eachcol(X) |> collect, us = eachcol(U) |> collect)
    end

    stack_trajectories(trajs)
end

"""
Utility for converting vertically concatenated state trajectories (i.e. [x^1, x^2]) into a matrix of the form
    [[x^1_1, x^2_1]', ... , [x^1_T, x^2_T]']
"""
function states_to_matrix(states, dynamics::ProductDynamics)
    single_traj_length = state_dim(dynamics.subsystems[1]) * horizon(dynamics) # assumes same dynamics for all players
    trajs = mortar(
        [states[(1:single_traj_length) .+ (i - 1) * single_traj_length] for i in 1:num_players(dynamics)],
        [single_traj_length for i in 1:num_players(dynamics)],
    )
    mapreduce(τ -> reshape(τ, (state_dim(dynamics.subsystems[1]), horizon(dynamics))), vcat, blocks(trajs))
end

"Utility for packing trajectory."
function pack_trajectory(traj)
    trajs = unstack_trajectory(traj)
    mapreduce(vcat, trajs) do τ
        vcat(reduce(vcat, τ.xs), reduce(vcat, τ.us))
    end
end

"Utility for blocking parameter vector"
function block_parameters(θ, n_players, n_couples, dynamics)
    θ_block  = BlockArray(θ, [n_players .* state_dim(dynamics.subsystems[1]), n_players .* 2, n_players .* 2, n_couples, n_couples, n_couples])

    mortar([
        θ_block[Block(1)], # initial position 
        BlockArray(θ_block[Block(2)], [2 for _ in 1:n_players]), # Goal positions
        BlockArray(θ_block[Block(3)], [2 for _ in 1:n_players]), # Cost weights 
        θ_block[Block(4)], # ω
        θ_block[Block(5)], # α0
        θ_block[Block(6)], # ρ
    ])
end

"Convert a TrajectoryGame to a ParametricGame."
function build_parametric_game(; game = setup_trajectory_game(), horizon = 10, n_players, n_couples)

    # Construct costs.
    function player_cost(τ, θ, player_index)
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        ts = Iterators.eachindex(xs)
        Iterators.map(xs, us, ts) do x, u, t
            game.cost.discount_factor^(t - 1) * game.cost.stage_cost(x, u, t, θ)[player_index]
        end |> game.cost.reducer
    end

    fs = [(τ, θ) -> begin
        θ_blocked = block_parameters(θ, n_players, n_couples, game.dynamics)
        player_cost(τ, θ_blocked, ii)
        end for ii in 1:n_players]
        
    # Dummy individual constraints.
    gs = [(τ, θ) -> [0] for _ in 1:n_players]
    hs = [(τ, θ) -> [0] for _ in 1:n_players]

    # Shared equality constraints.
    g̃ = (τ, θ) -> let
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        θ_blocked = block_parameters(θ, n_players, n_couples, game.dynamics)

        initial_state = θ_blocked[Block(1)]

        # Force all players to start at the given initial condition.
        g̃1 = xs[1] - initial_state

        # Dynamics constraints.
        ts = Iterators.eachindex(xs)
        g̃2 = mapreduce(vcat, ts[2:end]) do t
            xs[t] - game.dynamics(xs[t - 1], us[t - 1])
        end

        vcat(g̃1, g̃2)
    end

    # Shared inequality constraints.
    h̃ =
        (τ, θ) -> let
            (; xs, us) = unpack_trajectory(τ; game.dynamics)
            θ_blocked = block_parameters(θ, n_players, n_couples, game.dynamics)

            # Collision-avoidance constraint (hyperplanes in this case)
            h̃1 = game.coupling_constraints(xs, us, θ_blocked)

            # Actuator/state limits.
            actuator_constraint = TrajectoryGamesBase.get_constraints_from_box_bounds(
                control_bounds(game.dynamics),
            )
            h̃3 = mapreduce(vcat, us) do u
                actuator_constraint(u)
            end

            state_constraint =
                TrajectoryGamesBase.get_constraints_from_box_bounds(state_bounds(game.dynamics))
            h̃4 = mapreduce(vcat, xs) do x
                state_constraint(x)
            end

            vcat(h̃1, h̃3, h̃4)
        end

    ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = state_dim(game.dynamics) + n_players * (2 + 2) + 3 * n_couples,
        primal_dimensions = [
            horizon * (state_dim(game.dynamics, ii) + control_dim(game.dynamics, ii)) for ii in 1:n_players
        ],
        equality_dimensions = [1 for _ in 1:n_players],
        inequality_dimensions = [1 for _ in 1:n_players],
        shared_equality_dimension = state_dim(game.dynamics) +
                                    (horizon - 1) * state_dim(game.dynamics),
        shared_inequality_dimension = horizon * (
            n_couples + # make this change with number of hyperplane constraints 
            sum(isfinite.(control_bounds(game.dynamics).lb)) +
            sum(isfinite.(control_bounds(game.dynamics).ub)) +
            sum(isfinite.(state_bounds(game.dynamics).lb)) +
            sum(isfinite.(state_bounds(game.dynamics).ub))
        ),
    )
end

"Generate an initial guess for primal variables following a zero input sequence."
function generate_initial_guess(;
    game::TrajectoryGame{<:ProductDynamics},
    parametric_game::ParametricGame,
    horizon,
    initial_state,
)
    rollout_strategy =
        map(1:num_players(game)) do ii
            (x, t) -> zeros(control_dim(game.dynamics, ii))
        end |> TrajectoryGamesBase.JointStrategy

    zero_input_trajectory =
        TrajectoryGamesBase.rollout(game.dynamics, rollout_strategy, initial_state, horizon)

    vcat(
        pack_trajectory(zero_input_trajectory),
        zeros(total_dim(parametric_game) - sum(parametric_game.primal_dimensions)),
    )
end

"Solve a parametric trajectory game, where the parameter is just the initial state."
function TrajectoryGamesBase.solve_trajectory_game!(
    game::TrajectoryGame{<:ProductDynamics},
    horizon,
    θ,
    strategy;
    parametric_game = build_parametric_game(; game, horizon),
    verbose = false,
    solving_info = nothing,
)
    # Solve, maybe with warm starting.
    if !isnothing(strategy.last_solution) && strategy.last_solution.status == PATHSolver.MCP_Solved
        solution = solve(
            parametric_game,
            θ;
            initial_guess = strategy.last_solution.variables,
            verbose,
        )
    else
        solution = solve(
            parametric_game,
            θ;
            initial_guess = generate_initial_guess(; game, parametric_game, horizon, initial_state = θ[Block(1)]),
            verbose,
        )
    end

    if !isnothing(solving_info)
        push!(solving_info, solution.info)
    end

    # Update warm starting info.
    if solution.status == PATHSolver.MCP_Solved
        strategy.last_solution = solution
    end
    strategy.solution_status = solution.status

    # Pack solution into OpenLoopStrategy.
    trajs = unstack_trajectory(unpack_trajectory(mortar(solution.primals); game.dynamics))
    JointStrategy(map(traj -> OpenLoopStrategy(traj.xs, traj.us), trajs))
end

"Receding horizon strategy that supports warm starting."
Base.@kwdef mutable struct WarmStartRecedingHorizonStrategy
    game::TrajectoryGame
    parametric_game::ParametricGame
    receding_horizon_strategy::Any = nothing
    time_last_updated::Int = 0
    turn_length::Int
    horizon::Int
    last_solution::Any = nothing
    context_state::Any = nothing
    solution_status::Any = nothing
    parameters::Any = nothing
end

function (strategy::WarmStartRecedingHorizonStrategy)(state, time)
    plan_exists = !isnothing(strategy.receding_horizon_strategy)
    time_along_plan = time - strategy.time_last_updated + 1
    plan_is_still_valid = 1 <= time_along_plan <= strategy.turn_length

    θ = strategy.parameters

    update_plan = !plan_exists || !plan_is_still_valid
    if update_plan
        strategy.receding_horizon_strategy = TrajectoryGamesBase.solve_trajectory_game!(
            strategy.game,
            strategy.horizon,
            θ,
            strategy;
            strategy.parametric_game,
            verbose = true
        )
        strategy.time_last_updated = time
        time_along_plan = 1
    end

    strategy.receding_horizon_strategy(state, time_along_plan)
end

"Visualize a strategy `γ` on a makie canvas using the base color `color`."
function TrajectoryGamesBase.visualize!(
    canvas,
    γ::Makie.Observable{<:OpenLoopStrategy};
    color = :black,
    weight_offset = 0.0,
)
    Makie.series!(canvas, γ; color = [(color, min(1.0, 0.9 + weight_offset))])
end

function Makie.convert_arguments(::Type{<:Makie.Series}, γ::OpenLoopStrategy)
    traj_points = map(s -> Makie.Point2f(s[1:2]), γ.xs)
    ([traj_points],)
end

"2D satellite dynamics"
function cwh_satellite_2D(; dt = 5.0, n = 0.001, m = 100.0, kwargs...)
    # Layout is x := (px, py, vx, vy) and u := (ax, ay).
    time_invariant_linear_dynamics(;
    A = [ 4-3*cos(n*dt)         0  1/n*sin(n*dt)     2/n*(1-cos(n*dt))        ;
             6*(sin(n*dt)-n*dt) 1 -2/n*(1-cos(n*dt)) 1/n*(4*sin(n*dt)-3*n*dt) ;   
             3*n*sin(n*dt)      0  cos(n*dt)         2*sin(n*dt)              ;
            -6*n*(1-cos(n*dt))  0 -2*sin(n*dt)       4*cos(n*dt)-3            ],

    B = 1/m*[ 1/n^2(1-cos(n*dt))     2/n^2*(n*dt-sin(n*dt))       ;
             -2/n^2*(n*dt-sin(n*dt)) 4/n^2*(1-cos(n*dt))-3/2*dt^2 ;
              1/n*sin(n*dt)          2/n*(1-cos(n*dt))            ;
             -2/n*(1-cos(n*dt))      4/n*sin(n*dt)-3*dt           ;],
        kwargs...,
    )
end

"Visualize rotating hyperplanes"
function visualize_rotating_hyperplanes(
    states,
    parameters;
    title = "",
    filename = "",
    koz = true,
    hyperplanes = true,
    fps = 5,
    save_frame = nothing,
    noisy = false,
)

    # Parameters 
    text_size = 20
    text_size_ticks = 15
    marker_size_goals = 12
    marker_size_players = 7.5
    marker_observations = 4.0
    line_width = 4
    line_width_hyperplanes = 3

    # Useful stuff
    position_indices = vcat(
        [[1 2] .+ (player - 1) * parameters.n_states_per_player for player in 1:(parameters.n_players)]...,
    )
    couples = parameters.couples
    colors = palette(:default)[1:(parameters.n_players)]
    T = size(states,2)

    # Domain
    goals_x = [parameters.goals[player][1] for player in 1:(parameters.n_players)]
    goals_y = [parameters.goals[player][2] for player in 1:(parameters.n_players)]
    x_domain = extrema(hcat(states[position_indices[:, 1], :], goals_x))
    y_domain = extrema(hcat(states[position_indices[:, 2], :], goals_y))
    x_domain = 1.1 .* x_domain
    y_domain = 1.1 .* y_domain
    domain = [minimum([x_domain[1], y_domain[1]]), maximum([x_domain[2], y_domain[2]])]    

    θs = zeros(length(couples))
    player_couples = [findall(couple -> couple[1] == player_idx, couples) for player_idx in 1:parameters.n_players] 
    for player_idx in 1:parameters.n_players
        for (couple_idx, couple) in enumerate(couples[player_couples[player_idx]])
            parameter_idx = player_couples[player_idx][couple_idx]
            idx_ego   = (1:2) .+ (couple[1] - 1)*parameters.n_states_per_player
            idx_other = (1:2) .+ (couple[2] - 1)*parameters.n_states_per_player
            x_ego   = states[idx_ego,1]
            x_other = states[idx_other,1]
            x_diff  = x_ego - x_other
            θ = atan(x_diff[2], x_diff[1])

            θs[parameter_idx] = θ
        end
    end

    # Define useful vectors
    function n(t, θ, α, ω)
        [cos(θ + α + ω * (t - 1) * parameters.ΔT), sin(θ + α + ω * (t - 1) * parameters.ΔT)]
    end

    function n0(couple, states, position_indices)
        states[position_indices[couple[1], 1:2], 1] - states[position_indices[couple[2], 1:2], 1] 
    end

    # Animation of trajectory 
    anim = Plots.@animate for i = 1:T
        # Plot trajectories
        Plots.plot(;
            legend = false,
            title = title,
            xlabel = "x position [m]",
            ylabel = "y position [m]",
            yrotation = 90,
            size = (500, 500),
            guidefontsize = text_size,
            tickfontsize = text_size_ticks,
            linewidth = line_width,
        )
        if noisy
            Plots.scatter!(
                [states[position_indices[player, 1], 2:i] for player in 1:(parameters.n_players)],
                [states[position_indices[player, 2], 2:i] for player in 1:(parameters.n_players)],
                markersize = marker_observations,
                msw = 0,
                label = "",
            )
        else
            Plots.plot!(
                [states[position_indices[player, 1], 2:i] for player in 1:(parameters.n_players)],
                [states[position_indices[player, 2], 2:i] for player in 1:(parameters.n_players)],
                linewidth = line_width,
                label = "",
            )
        end

        # Scatter transparent goals for start position 
        Plots.scatter!(
            [states[position_indices[player, 1], 1] for player in 1:(parameters.n_players)],
            [states[position_indices[player, 2], 1] for player in 1:(parameters.n_players)],
            markersize = marker_size_goals,
            marker = :star4,
            markercolor=:white,
            markerstrokecolor=colors,
            label = "Start position",
        )

        # plot goals from parameters info with an x
        Plots.scatter!(
            [parameters.goals[player][1] for player in 1:(parameters.n_players)],
            [parameters.goals[player][2] for player in 1:(parameters.n_players)],
            markersize = marker_size_goals,
            marker = :star4,
            color = colors,
            msw = 0,
            label = "Goal position",
        )

        # Plot KoZs and hyperplanes
        for (couple_idx, couple)  in enumerate(couples)
            if koz
                # Plot KoZs around hyperplane owner
                Plots.plot!(
                    [states[position_indices[couple[2], 1], i] + parameters.ρs[couple_idx] * cos(θ) for θ in range(0,stop=2π,length=100)], 
                    [states[position_indices[couple[2], 2], i] + parameters.ρs[couple_idx] * sin(θ) for θ in range(0,stop=2π,length=100)], 
                    color = colors[couple[1]], 
                    legend = false,
                    fillalpha = 0.25,
                    fill = true,
                    # linewidth = line_width_hyperplanes,
                    linewidth = 0.0,
                    label = "Keep-out zone",
                )
            end
            # Plot hyperplane normal
            ni =
                parameters.ρs[couple_idx] *
                n(i, θs[couple_idx], parameters.α0s[couple_idx], parameters.ωs[couple_idx])
            # Plots.plot!(
            #     [states[position_indices[couple[2], 1], i], states[position_indices[couple[2], 1], i] + ni[1]],
            #     [states[position_indices[couple[2], 2], i], states[position_indices[couple[2], 2], i] + ni[2]],
            #     arrow = true,
            #     color = colors[couple[1]],
            # )
            # Plot hyperplane 
            if hyperplanes
                hyperplane_domain = 10*range(domain[1],domain[2],100)
                p = states[position_indices[couple[2], 1:2], i] + ni
                Plots.plot!(hyperplane_domain .+ p[1],
                    [-ni[1]/ni[2]*x + p[2] for x in hyperplane_domain],
                    color = colors[couple[1]],
                    linewidth = line_width_hyperplanes,
                    linestyle = :dot,
                    label = "Hyperplane",
                )
            end
        end

        # Plot player positions on top 
        Plots.scatter!(
            [states[position_indices[player, 1], i] for player in 1:(parameters.n_players)],
            [states[position_indices[player, 2], i] for player in 1:(parameters.n_players)],
            markersize = marker_size_players,
            msw = 3,
            color = colors,
            label = "Robot position",
        )

        # Set domain
        Plots.plot!(xlims = domain, ylims = domain)

        # Annotate time at top right 
        Plots.annotate!((
            x_domain[2] - (x_domain[2] - x_domain[1]) / 10,
            y_domain[2] - (y_domain[2] - y_domain[1]) / 30,
            (string("t = ", trunc(Int, i * parameters.ΔT), "s"), text_size),
        ))

        # Save if at saveframe
        if !isnothing(save_frame) && i == save_frame
            # Plots.plot!(legend = :outerright, legendfontsize=text_size)
            Plots.savefig("figures/"*filename*"_frame.png")
        end
    end
    Plots.gif(anim, fps = fps, "figures/"*filename*".gif")
end    

"Solve the forward game"
function forward(θ, game_setup; visualize = false, filename = "forward")
    @unpack game, parametric_game, dt, couples = game_setup
    
    # Simulate forward
    solution = solve(
            parametric_game,
            θ; 
        verbose = false,
        return_primals = true
        )

    # Extract states, reshape as a matrix and save in a tuple
    trajectory = (;
        x = states_to_matrix(
            extract_states(
                solution.variables[1:sum(parametric_game.primal_dimensions)],
                game.dynamics,
            ),
            game.dynamics,
        )
    )

    # Plot hyperplanes
    if visualize
        plot_parameters = 
            (;
                n_players = num_players(game),
                n_states_per_player = state_dim(game.dynamics.subsystems[1]),
                goals = [θ[Block(2)][Block(player)] for player in 1:num_players(game)],
                couples,
                ωs = θ[Block(4)],
                α0s = zeros(length(couples)),
                ρs = θ[Block(6)],
                ΔT = dt
            )
        visualize_rotating_hyperplanes(
            trajectory.x,
            plot_parameters;
            # title = string(n_players)*"p",
            koz = true, 
            hyperplanes = true,
            fps = 10.0,
            filename,
            save_frame = 17,
        )
    end

    solution
end

"Setup parameteres for the forward game, inverse game, and the MC analysis parameters"
function setup_experiment(;n_players = 2)

    function unitvector(θ)
        [cos(θ), sin(θ)]
    end

    # ---- FORWARD GAME PARAMETERS ----

    # Game parameters
    horizon = 22
    dt = 10.0
    scale = 100.0
    initial_state = mortar([vcat(-scale .* unitvector(pi/n_players*(i-1)), [0.0,0.0]) for i in 1:n_players])
    goals = mortar([scale .* unitvector(pi/n_players*(i-1)) for i in 1:n_players])
    couples = [(i, j) for i in 1:n_players for j in i+1:n_players]
    ωs = [0.015 for _ in couples]
    # ωs = [0.015]
    α0s = [
        atan(
            initial_state[Block(couple[1])][2] - initial_state[Block(couple[2])][2],
            initial_state[Block(couple[1])][1] - initial_state[Block(couple[2])][1],
        ) for couple in couples
    ]
    # ρs = [30.0]
    ρs = [20.0 for _ in couples]
    weights = mortar([[10.0, 0.0001] for _ in 1:n_players])
    m   = 100.0 # kg
    r₀ = (400 + 6378.137) # km
    grav_param  = 398600.4418 # km^3/s^2
    n = sqrt(grav_param/(r₀^3)) # rad/s   

    # Ground truth parameters 
    θ_truth = mortar([initial_state, goals, weights, ωs, α0s, ρs])

    # Set up games
    game = setup_trajectory_game(;n_players, horizon, dt, n, m, couples)
    parametric_game = build_parametric_game(; game, horizon, n_players, n_couples = length(couples))

    # ---- INVERSE GAME PARAMETERS ----

    # Step parameters
    learning_rate_x0_pos = 10
    learning_rate_x0_vel = 1e-2
    learning_rate_ω = 5e-4
    learning_rate_ρ = 1.0
    learning_parameters = (; learning_rate_x0_pos, learning_rate_x0_vel, learning_rate_ω, learning_rate_ρ)

    # Initial parameter guess TODO remove this. Shouldn't be part of the setup struct 
    θ_guess = mortar([
        initial_state,
        goals,
        weights,
        [0.008 for _ in couples], # From cited paper
        α0s,
        [10.0 for _ in couples]
    ])
    
    # ---- MONTE CARLO PARAMETERS ----
    rng = MersenneTwister(1234)

    # ---- Pack everything ----
    (;game, parametric_game, dt, θ_guess, θ_truth, couples, learning_parameters, rng)
end

"Loss function. Norm square of difference between observed and predicted primals"
function inverse_loss(observation, θ, game, parametric_game; initial_guess = nothing)

    # Predicted equilibrium 
    solution_predicted = solve(
            parametric_game,
            θ;
        initial_guess, 
        verbose = false,
        return_primals = false
        )

    loss = norm_sqr(
        extract_states(
            solution_predicted.variables[1:sum(parametric_game.primal_dimensions)],
            game.dynamics,
        ) .- observation,
    )

    # Norm sqr of difference between predicted and observed (states only)
    solution_predicted.variables,
    solution_predicted.status,
    loss
end

"Compose θ_guess from models"
function compose_from_models(models, indices, θ_guess)
    block_dict = Dict(
        :x0 => Block(1),
        :v0 => Block(1),
        :goals => Block(2),
        :weights => Block(3),
        :ω => Block(4),
        :α0 => Block(5),
        :ρ => Block(6),
    )

    # Extract parameters from model
    for model in models
        for (key, value) in pairs(model)
            if haskey(indices, key)
                θ_guess[block_dict[key]][indices[key]] = value
            else
                θ_guess[block_dict[key]] = value
            end
        end
    end

    θ_guess
end

"""
Solve the inverse game taking gradient steps

Observation is a vector of concatenated states for all players i.e. [x^1, ... , x^n_players]
"""
function inverse(observation, θ_guess, game_setup; max_grad_steps = 10, tol = 1e-1, verbose = false)

    @unpack game, parametric_game, θ_truth, learning_parameters = game_setup
    @unpack learning_rate_x0_pos, learning_rate_x0_vel, learning_rate_ω, learning_rate_ρ = learning_parameters

    # Setup indices
    indices = Dict(
        :x0 => reduce(
            vcat,
            [
                (1:2) .+ (i - 1) * state_dim(game.dynamics.subsystems[1]) for
                i in 1:num_players(game)
            ],
        ),
        :v0 => reduce(
            vcat,
            [
                (3:4) .+ (i - 1) * state_dim(game.dynamics.subsystems[1]) for
                i in 1:num_players(game)
            ],
        ),
    )

    # Setup optimiser
    model_x0 = (x0 = copy(θ_guess[Block(1)])[indices[:x0]],)
    model_v0 = (v0 = copy(θ_guess[Block(1)])[indices[:v0]],)
    model_ω = (ω = copy(θ_guess[Block(4)]),)
    model_ρ = (ρ = copy(θ_guess[Block(6)]),)

    # Setup chain 
    state_tree_x0 = Optimisers.setup(Optimisers.Adam(learning_rate_x0_pos, (0.9, 0.999)), model_x0)
    state_tree_v0 = Optimisers.setup(Optimisers.Adam(learning_rate_x0_vel, (0.8, 0.999)), model_v0)
    state_tree_ω = Optimisers.setup(Optimisers.Adam(learning_rate_ω, (0.8, 0.999)), model_ω)
    state_tree_ρ = Optimisers.setup(Optimisers.Adam(learning_rate_ρ, (0.8, 0.999)), model_ρ)

    # Print first step 
    z_new, status_first, loss_first = inverse_loss(observation, θ_guess, game, parametric_game)

    # Print hyperplane parameters horizontally
    verbose && println(
        "0: ",
        "Δx0: ",
        norm(θ_guess[Block(1)][indices[:x0]] - θ_truth[Block(1)][indices[:x0]]),
        ", Δv0: ",
        norm(θ_guess[Block(1)][indices[:v0]] - θ_truth[Block(1)][indices[:v0]]),
        ", ωs: ",
        trunc.(θ_guess[Block(4)], digits = 3),
        ", ρs: ",
        trunc.(θ_guess[Block(6)], digits = 3),
        " L:  ",
        loss_first,
    )

    # Break if first step did not converge
    if status_first != PATHSolver.MCP_Solved
        verbose && println( "   Stopping: First step did not converge.") 
        return false, θ_guess
    end

    # Gradient wrt hyperplane parameters only 
    grad_norms  = []
    grad_norms_x0 = Float64[]
    grad_norms_v0 = Float64[]
    grad_norms_ω = Float64[]
    grad_norms_ρ = Float64[]
    losses = []
    θ_guess = copy(θ_guess)
    for i in 1:max_grad_steps

        # Gradient 
        grad = Zygote.gradient(
            θ -> inverse_loss(observation, θ, game, parametric_game; initial_guess = z_new)[3],
            θ_guess,
        )[1]
        grad_block = BlockArray(grad, blocksizes(θ_guess)[1])

        grad_norm_x0 = norm(grad_block[Block(1)][indices[:x0]])
        grad_norm_v0 = norm(grad_block[Block(1)][indices[:v0]])
        grad_norm_ω = norm(grad_block[Block(4)])
        grad_norm_ρ = norm(grad_block[Block(6)])
        grad_norm = norm([grad_block[Block(1)], grad_block[Block(4)], grad_block[Block(6)]])

        # Update models
        state_tree_x0, model_x0 = Optimisers.update!(state_tree_x0, model_x0, (x0 = grad_block[Block(1)][indices[:x0]],))
        state_tree_v0, model_v0 = Optimisers.update!(state_tree_v0, model_v0, (v0 = grad_block[Block(1)][indices[:v0]],))
        state_tree_ω, model_ω  = Optimisers.update!(state_tree_ω, model_ω, (ω = grad_block[Block(4)],))
        state_tree_ρ, model_ρ  = Optimisers.update!(state_tree_ρ, model_ρ, (ρ = grad_block[Block(6)],))

        # New parameters 
        θ_guess = compose_from_models([model_x0, model_v0, model_ω, model_ρ], indices, copy(θ_guess))

        # New solution 
        z_new, status_new, loss_new = inverse_loss(observation, θ_guess, game, parametric_game; initial_guess = z_new)

        # Break if new step did not converge
        if status_new != PATHSolver.MCP_Solved
            verbose && println("    Stopping: New step did not converge.") 
            return false, θ_guess
        end

        # Break if any of the radii is negative 
        if any(θ_guess[Block(6)] .< 0)
            verbose && println("    Stopping: Negative radius.") 
            return false, θ_guess
        end

        # Print
        verbose && println(
            i,
            ": Δx0: ",
            trunc(norm(θ_guess[Block(1)][indices[:x0]] - θ_truth[Block(1)][indices[:x0]]), digits = 2),
            ", Δv0: ",
            trunc(norm(θ_guess[Block(1)][indices[:v0]] - θ_truth[Block(1)][indices[:v0]]), digits = 2),
            ": ωs: ",
            trunc.(θ_guess[Block(4)], digits = 3),
            ", ρs: ",
            trunc.(θ_guess[Block(6)], digits = 3),
            ", ∇L: ",
            trunc.(grad_norm, digits = 3),
            ", L: ",
            loss_new,
        )
        push!(grad_norms_x0, grad_norm_x0)
        push!(grad_norms_v0, grad_norm_v0)
        push!(grad_norms_ω, grad_norm_ω)
        push!(grad_norms_ρ, grad_norm_ρ)
        push!(grad_norms, grad_norm)
        push!(losses, loss_new)

        # Break if grad norm is small enough
        if grad_norm < tol
            # Print momentum norm and break 
            verbose && println("Stopping: Grad norm = ", grad_norm, " < ", tol) 
            break
        end
    end

    # Print final guess
    verbose && println("Final hyperplane parameters: ", θ_guess[Block(4)])

    # # Plot losses and gradient norm 
    # fig_loss = Plots.plot(1:length(losses), losses, label = "loss", yaxis=:log)
    # Plots.display(fig_loss)

    # Plot grad_norms_x0, grad_norms_v0, grad_norms_ω, grad_norms_ρ in a single plot with subplots
    # fig_x0 = Plots.plot(1:length(grad_norms_x0), grad_norms_x0, ylims = (0,Inf), label = "|∇L_x0|", yaxis=:log)
    # fig_v0 = Plots.plot(1:length(grad_norms_v0), grad_norms_v0, ylims = (0,Inf), label = "|∇L_v0|", yaxis=:log)
    # fig_ω = Plots.plot(1:length(grad_norms_ω), grad_norms_ω, ylims = (0,Inf), label = "|∇L_ω|", yaxis=:log)
    # fig_ρ = Plots.plot(1:length(grad_norms_ρ), grad_norms_ρ, ylims = (0,Inf), label = "|∇L_ρ|", yaxis=:log)
    # fig = Plots.plot(fig_x0, fig_v0, fig_ω, fig_ρ, layout = (2,2), legend = true)
    # Plots.display(fig)

    # # Plot hyperplanes with final trajectory
    # θ_inverse = copy(game_setup.θ_truth)
    # θ_inverse[Block(4)] .= θ_guess[Block(4)]
    # θ_inverse[Block(6)] .= θ_guess[Block(6)]
    # forward(θ_inverse, game_setup; visualize = true, filename = "inverse")

    # Return parameters
    return true, θ_guess

end

"""
Utility to extract states from the primals of an MCP solution.

Primals should be in the form of a single vector. 
"""
function extract_states(primals, dynamics::ProductDynamics)
    n_players = num_players(dynamics)
    n_states_per_player = state_dim(dynamics.subsystems[1])
    n_controls_per_player = control_dim(dynamics.subsystems[1])
    T = horizon(dynamics)

    primals[reduce(
        vcat,
        [
            (1:(T * n_states_per_player)) .+
            (n_player - 1) * T * (n_states_per_player + n_controls_per_player) for
            n_player in 1:n_players
        ],
    ),]
end

"Run Monte Carlo simulation for a sequence of noise levels"
function mc(trials, game_setup, solution_forward; kwargs...)

    @unpack game, parametric_game, θ_guess, θ_truth, rng = game_setup

    # ---- Useful functions/numbers ----
    n_players = num_players(game)
    n_states_per_player = state_dim(game.dynamics.subsystems[1])
    n_controls_per_player = control_dim(game.dynamics.subsystems[1])
    T = horizon(game.dynamics)

    "Compute trajectory reconstruction error. See Peters et al. 2021 experimental section"
    function compute_rec_error(primals_forward, primals_inverse)

        states_forward = extract_states(primals_forward, game.dynamics)
        states_inverse = extract_states(primals_inverse, game.dynamics)

        # Position indices 
        position_indices = (player, t) -> (1:2) .+ (player-1)*T*n_states_per_player .+ (t-1)*n_states_per_player

        # Sum of norms
        reconstruction_error = 0
        for player in 1:n_players
            for t in 1:horizon(game.dynamics)
                reconstruction_error += norm(
                    states_forward[position_indices(player, t)] -
                    states_inverse[position_indices(player, t)],
                )
            end
        end

        1/(n_players * horizon(game.dynamics)) * reconstruction_error
    end

    # ---- Noise levels ----
    # noise_levels = 0.0:0.1:2.5
    # noise_levels = 20.0
    noise_levels = 0.0:1.0:20.0

    # ---- Monte Carlo ----
    println("Starting Monte Carlo for ", length(noise_levels), " noise levels and ", trials, " trials each.")
    println("True parameters: ", θ_truth[Block(4)][1]," ", θ_truth[Block(6)][1])
    
    # Initialize results array 1x6 empty array of floats
    results = zeros(Float64, 1, 6)

    # Setup solution as a matrix
    primals_forward = solution_forward.variables[1:sum(parametric_game.primal_dimensions)]
    states_forward = extract_states(primals_forward, game.dynamics)

    # Copy guess 
    θ_guess = copy(θ_guess)
    for noise_level in noise_levels
        println("Noise level: ", noise_level)

        for i in 1:trials
            # Extract states only and add noise
            observation =
                states_forward + noise_level * randn(rng, T * n_states_per_player * n_players)

            # Initialize result vector with NaN
            result = ones(Float64, 1, 6) * NaN

            # Guess initial position as first observed position with zero velocity
            x0_indices = reduce(vcat, [(1:4) .+ (player-1)*T*n_states_per_player for player in 1:n_players])
            θ_guess[Block(1)] = BlockArray(observation[x0_indices], [n_states_per_player for player in 1:n_players])
            vel_indices = reduce(vcat, [(3:4) .+ (player-1)*n_states_per_player for player in 1:n_players])
            θ_guess[vel_indices] .= 0.0

            # Solve inverse game with guessed parameter
            converged_inverse, inverse_parameters = inverse(observation, θ_guess, game_setup; kwargs...)

            # Compute trajectory for inverse game only if converged 
            if converged_inverse 
                inverse_elapsed = @elapsed solution_inverse = solve(
                    parametric_game,
                    inverse_parameters;
                    initial_guess = nothing, 
                    verbose = false,
                    return_primals = false
                )

                reconstruction_error = compute_rec_error(primals_forward, solution_inverse.variables[1:sum(parametric_game.primal_dimensions)])
            else
                inverse_elapsed = NaN
                reconstruction_error = NaN
            end

            # Assemble result matrix
            result = [
                noise_level,
                converged_inverse ? 1.0 : 0.0,
                inverse_parameters[Block(4)][1],
                inverse_parameters[Block(6)][1],
                reconstruction_error,
                inverse_elapsed,
            ]

            # Append result to results array
            results = vcat(results, result')
            
            # Print progress 
            println("   ",
                lpad(string(noise_level), 6, "0"),
                " Trial: ",
                i,
                "/",
                trials,
                " Converged: ",
                round(result[2], digits = 1),
                " ω: ",
                round(result[3], digits = 3),
                " ρ: ",
                round(result[4], digits = 3),
                " Error: ",
                round(result[5], digits = 5),
                " Time: ",
                round(result[6], digits = 3),
            )

            # Stop if first noise level 
            if noise_level == noise_levels[1] && i == 2
                break
            end
        end

        # Dirty hack 
        if noise_level == noise_levels[1]
            results = results[2:end, :]
        end

        idx_current = results[:, 1] .== noise_level
        idx_converged = (results[:, 1] .== noise_level ) .* (results[:, 2] .== 1.0)
        num_converged = sum(idx_converged)

        println(
            "Convergence rate: ",
            num_converged / trials,
            " error = ",
            sum(results[idx_converged, 5]) / num_converged,
            " time = ",
            sum(results[idx_converged, 6]) / num_converged,
        )

        num_converged == 0 && @warn "No convergence for noise level ", noise_level, ". Plotting will fail."
    end 

    # ---- Save results ----
    df = DataFrame(
        noise_level = results[:, 1],
        converged = results[:, 2],
        ω = results[:, 3],
        ρ = results[:, 4],
        reconstruction_error = results[:, 4],
        time = results[:, 6],
    )
    CSV.write("mc_noise.csv", df, header = false)

    try
        plotmc(results, noise_levels, game_setup)
    catch
        println("Plotting failed")
    end
    
    return results, noise_levels
end

"Given a set of parameters, run Monte Carlo simulation for different initial velocities"
function mc_inits(trials; n_players_vec = 2:1:5, velocity_σs = 0:0.1:1.0, convergence_threshold = 0.9)

all_convergence_rates = Vector{Vector{Float64}}()
all_velocity_σs = []
for n_players in n_players_vec

    setup_time = @elapsed game_setup = setup_experiment(;n_players)
    @unpack game, parametric_game, rng = game_setup
    println("---- ", n_players, "-player game. Setup time = ", setup_time, " ----")

    # All velocity indices in a single vector
    velocity_indices = reduce(
        vcat,
        [
            (3:4) .+ (player - 1) * state_dim(game.dynamics.subsystems[1]) for
            player in 1:n_players
        ],
    )

    convergence_rates = Float64[]
    convergence_rate_flag = false
    for velocity_σ in velocity_σs
        converged = Bool[]
        for trial in 1:trials
            # Sample initial state with perturbed velocity 
            velocity_disturbance = velocity_σ * randn(rng, 2 * n_players)
            θ_sample = copy(game_setup.θ_truth)
            θ_sample[Block(1)][velocity_indices] += velocity_disturbance

            # Run forward game with given parameters
            solution = forward(θ_sample, game_setup; visualize = false)    

            # Print and save 
            println("   v_σ = ", velocity_σ, " trial ", trial, " converged ", solution.status == PATHSolver.MCP_Solved ? "true" : "false")
            solution.status == PATHSolver.MCP_Solved ? push!(converged, true) : push!(converged, false)
        end 

        # Print convergence rate and save 
        println("   v_σ = ", velocity_σ, " convergence rate ", sum(converged) / trials)
        push!(convergence_rates, sum(converged) / trials)

        if sum(converged) / trials < convergence_threshold
            println("   Convergence rate too low. Stopping.")
            convergence_rate_flag = true
            break
        end
    end

    push!(all_convergence_rates, convergence_rates)
    if convergence_rate_flag
        push!(all_velocity_σs, velocity_σs[1:length(convergence_rates)])
    else
        push!(all_velocity_σs, velocity_σs)
    end

    println(n_players, "-player game. Convergence rates: ", convergence_rates, " velocity_σs: ", all_velocity_σs[end],"\n")
end

Main.@infiltrate

# Parameters 
text_size = 23
line_width = 4
colors = palette(:default)[1:(length(all_velocity_σs))]
Makie.set_theme!()

fig_convergence = Makie.Figure(resolution = (800, 300), fontsize = text_size)
ax_convergence = Makie.Axis(
    fig_convergence[1, 1],
    xlabel = "Velocity noise standard deviation [m/s]",
    ylabel = "Convergence %",
    limits = ((0, all_velocity_σs[1][end]), (0, 110)),
)
for (σs, rates, player_i) in zip(all_velocity_σs, all_convergence_rates, 1:length(all_velocity_σs))
        Makie.lines!(
            ax_convergence,
            σs,
            100 .* rates,
            color = colors[player_i],
            label = string(player_i+1) * (" robots"),
            linewidth = line_width,
        )
end
Makie.axislegend(position = :lb)
Makie.save("figures/mc_init_convergence.jpg", fig_convergence)

nothing

end

"Plot convergence rate, average reconstruction error, and parameter error vs noise level"
function plotmc(results, noise_levels, game_setup)

    "Compute the interquartile range of a sample. 
    Taken from https://turreta.com/blog/2020/03/28/find-interquartile-range-in-julia/"
    function iqr(samples)
        samples = sort(samples)

        # Get the size of the samples
        samples_len = length(samples)

        # Divide the size by 2
        sub_samples_len = div(samples_len, 2)

        # Know the indexes
        start_index_of_q1 = 1
        end_index_of_q1 = sub_samples_len
        start_index_of_q3 = samples_len - sub_samples_len + 1
        end_index_of_q3 = samples_len

        # Q1 median value
        median_value_of_q1 = median(view(samples, start_index_of_q1:end_index_of_q1))

        # Q2 median value
        median_value_of_q3 = median(view(samples, start_index_of_q3:end_index_of_q3))

        # Find the IQR value
        iqr_result = median_value_of_q3 - median_value_of_q1
        return iqr_result
    end

    # Parameters
    color_iqr = :dodgerblue
    Makie.set_theme!()
    text_size = 23

    # Create makie screens 

    # Calculation
    idx_converged = results[:,2] .== 1.0
    trials = sum(results[:, 1] .== noise_levels[2])

    # Plot convergence rate  
    fig_convergence = Makie.Figure(resolution = (800, 230), fontsize = text_size)
    ax_convergence = Makie.Axis(
        fig_convergence[1, 1],
        xlabel = "Noise standard deviation [m]",
        ylabel = "Convergence %",
        # title = "Convergence rate vs noise level",
        limits = (nothing, (0, 100)),

    )
    Makie.barplot!(
        ax_convergence,
        noise_levels,
        [sum(results[results[:, 1] .== noise_level, 2]) / (noise_level == noise_levels[1] ? 2.0 : trials) * 100 for noise_level in noise_levels],
    )
    Makie.rowsize!(fig_convergence.layout, 1, Makie.Aspect(1,0.2))

    # Plot bands for ω
    ω_median = [median(results[(results[:, 1] .== noise_level) .* idx_converged, 3]) for noise_level in noise_levels]
    ρ_median = [median(results[(results[:, 1] .== noise_level) .* idx_converged, 4]) for noise_level in noise_levels]

    ω_iqr = [iqr(results[(results[:, 1] .== noise_level) .* idx_converged, 3]) for noise_level in noise_levels]
    ρ_iqr = [iqr(results[(results[:, 1] .== noise_level) .* idx_converged, 4]) for noise_level in noise_levels]

    fig_bands = Makie.Figure(resolution = (800, 350), fontsize = text_size)
    ax_ω = Makie.Axis(
        fig_bands[1, 1],
        xlabel = "Noise standard deviation",
        ylabel = "ω [rad/s]",
        limits = ((noise_levels[1], noise_levels[end]), game_setup.θ_truth[Block(4)][1] > 0 ? (0, 2*game_setup.θ_truth[Block(4)][1]) : (2*game_setup.θ_truth[Block(4)][1], 0)),
    )
    Makie.scatter!(ax_ω, noise_levels, ω_median, color = color_iqr)
    Makie.band!(ax_ω, noise_levels, ω_median .- ω_iqr/2, ω_median .+ ω_iqr/2, color = (color_iqr, 0.2))
    Makie.hlines!(ax_ω, game_setup.θ_truth[Block(4)][1], color = color_iqr, linewidth = 2, linestyle = :dot)
    ax_ρ = Makie.Axis(
        fig_bands[1, 2],
        xlabel = "Noise standard deviation",
        ylabel = "ρ [m]",
        limits = ((noise_levels[1], noise_levels[end]), (0.5*game_setup.θ_truth[Block(6)][1], 1.5*game_setup.θ_truth[Block(6)][1])),
    )
    Makie.scatter!(ax_ρ, noise_levels, ρ_median, color = color_iqr)
    # Makie.band!(ax_ρ, noise_levels, clamp.(ρ_median .- ρ_iqr/2, game_setup.ρmin, Inf), ρ_median .+ ρ_iqr/2, color = (color_iqr, 0.2))
    Makie.band!(ax_ρ, noise_levels, ρ_median .- ρ_iqr/2, ρ_median .+ ρ_iqr/2, color = (color_iqr, 0.2))
    Makie.hlines!(ax_ρ, game_setup.θ_truth[Block(6)][1], color = color_iqr, linewidth = 2, linestyle = :dot)
    Makie.rowsize!(fig_bands.layout, 1, Makie.Aspect(1,0.9))
    

    # Plot reconstruction error
    reconstruction_error_median = [median(results[(results[:, 1] .== noise_level) .* idx_converged, 5]) for noise_level in noise_levels]
    reconstruction_error_iqr = [iqr(results[(results[:, 1] .== noise_level) .* idx_converged, 5]) for noise_level in noise_levels]
    fig_error = Makie.Figure(resolution = (800, 250), fontsize = text_size)
    ax_error = Makie.Axis(
        fig_error[1, 1],
        xlabel = "Noise standard deviation [m]",
        ylabel = "Reconstruction error [m]",
        # title = "Reconstruction error vs noise level",
        limits = ((noise_levels[1],noise_levels[end]),(0.0,nothing)),
    )
    Makie.scatter!(ax_error, noise_levels, reconstruction_error_median, color = color_iqr)
    Makie.band!(ax_error, noise_levels, clamp.(reconstruction_error_median .- reconstruction_error_iqr/2, 0, Inf), reconstruction_error_median .+ reconstruction_error_iqr/2, color = (color_iqr, 0.2))
    Makie.rowsize!(fig_error.layout, 1, Makie.Aspect(1,0.2))

    # # Save figures 
    Makie.save("figures/mc_noise_convergence.jpg", fig_convergence)
    Makie.save("figures/mc_noise_bands.jpg", fig_bands)
    Makie.save("figures/mc_noise_error.jpg", fig_error)

    return nothing
end