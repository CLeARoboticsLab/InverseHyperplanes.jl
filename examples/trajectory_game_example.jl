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
    rollout
using TrajectoryGamesExamples: planar_double_integrator, animate_sim_steps
using BlockArrays: mortar, blocks, BlockArray, Block
using GLMakie: GLMakie
using Makie: Makie
using PATHSolver: PATHSolver
using LinearAlgebra: norm_sqr, norm
using ProgressMeter: ProgressMeter
using Plots

"Utility to set up a (two player) trajectory game."
function setup_trajectory_game(; n_players = 2, dt = 5.0, n = 0.001, m = 100.0, couples = nothing)

    cost = let
        function stage_cost(x, u, t, θ)
            x1, x2 = blocks(x)
            u1, u2 = blocks(u)
            goals =   θ[Block(2)]
            weights = θ[Block(3)]

            # Both players want to minimize distance to goal state whil minimizing control effort. 
            [
                weights[Block(1)][1] * norm_sqr(x1[1:2] - goals[Block(1)]) + weights[Block(1)][2] * norm_sqr(u1),
                weights[Block(2)][1] * norm_sqr(x2[1:2] - goals[Block(2)]) + weights[Block(2)][2] * norm_sqr(u2),
            ]
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
        constraints = [
            vcat(
                map(
                    (tup) -> begin
                        (ii, x) = tup
                        x_ego = x[Block(couple[1])]
                        x_other = x[Block(couple[2])]

                        # Hyperplane normal 
                        n_arg = α0s[couple_idx] + ωs[couple_idx] * (ii - 1) * dt
                        n = [cos(n_arg), sin(n_arg)]

                        # Intersection of hyperplane w/ KoZ
                        p = x_other[1:2] + ρs[couple_idx] * n

                        # Hyperplane constraint
                        n' * (x_ego[1:2] - p)
                    end,
                    enumerate(xs),
                ),
            ) for (couple_idx, couple) in enumerate(couples)
        ]
        
        vcat(constraints...)  
    end

    agent_dynamics = cwh_satellite_2D(;
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-1.0, -1.0], ub = [1.0, 1.0]),
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

"Utility for packing trajectory."
function pack_trajectory(traj)
    trajs = unstack_trajectory(traj)
    mapreduce(vcat, trajs) do τ
        vcat(reduce(vcat, τ.xs), reduce(vcat, τ.us))
    end
end

"Utility for blocking parameter vector"
function block_parameters(θ, N, n_couples, dynamics)
    θ_block  = BlockArray(θ, [N .* state_dim(dynamics.subsystems[1]), N .* 2, N .* 2, n_couples, n_couples, n_couples])

    mortar([
        θ_block[Block(1)],
        BlockArray(θ_block[Block(2)], [2, 2]),
        BlockArray(θ_block[Block(3)], [2, 2]),
        θ_block[Block(4)],
        θ_block[Block(5)],
        θ_block[Block(6)],
    ])
end

"Convert a TrajectoryGame to a ParametricGame."
function build_parametric_game(; game = setup_trajectory_game(), horizon = 10, N = 2, n_couples)

    # Construct costs.
    function player_cost(τ, θ, player_index)
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        ts = Iterators.eachindex(xs)
        Iterators.map(xs, us, ts) do x, u, t
            game.cost.discount_factor^(t - 1) * game.cost.stage_cost(x, u, t, θ)[player_index]
        end |> game.cost.reducer
    end

    # fs = [(τ, θ) -> player_cost(τ, θ, ii) for ii in 1:N]
    fs = [(τ, θ) -> begin
        θ_blocked = block_parameters(θ, N, n_couples, game.dynamics)
        player_cost(τ, θ_blocked, ii)
        end for ii in 1:N]

    # Dummy individual constraints.
    gs = [(τ, θ) -> [0] for _ in 1:N]
    hs = [(τ, θ) -> [0] for _ in 1:N]

    # Shared equality constraints.
    g̃ = (τ, θ) -> let
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        θ_blocked = block_parameters(θ, N, n_couples, game.dynamics)

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
            θ_blocked = block_parameters(θ, N, n_couples, game.dynamics)

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
            # vcat(h̃3, h̃4)
        end

    ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = state_dim(game.dynamics) + N * (2 + 2) + 3 * n_couples,
        primal_dimensions = [
            horizon * (state_dim(game.dynamics, ii) + control_dim(game.dynamics, ii)) for ii in 1:N
        ],
        equality_dimensions = [1 for _ in 1:N],
        inequality_dimensions = [1 for _ in 1:N],
        shared_equality_dimension = state_dim(game.dynamics) +
                                    (horizon - 1) * state_dim(game.dynamics),
        shared_inequality_dimension = horizon * (
            1 + # make this change with number of hyperplane constraints 
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
    θ[Block(1)] = state

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

function visualize_rotating_hyperplanes(states, parameters; title = "", filename = "", koz = true, fps = 5, save_frame = nothing, noisy = false)

    # Parameters 
    text_size = 20
    text_size_ticks = 15
    marker_size_goals = 10
    marker_size_players = 7.5
    marker_observations = 4.0
    line_width = 4
    line_width_hyperplanes = 3

    # Useful stuff
    position_indices = vcat(
        [[1 2] .+ (player - 1) * parameters.n_states_per_player for player in 1:(parameters.n_players)]...,
    )
    couples = findall(parameters.adjacency_matrix)
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
                [states[position_indices[player, 1], 1:i] for player in 1:(parameters.n_players)],
                [states[position_indices[player, 2], 1:i] for player in 1:(parameters.n_players)],
                markersize = marker_observations,
            )
        else
            Plots.plot!(
                [states[position_indices[player, 1], 1:i] for player in 1:(parameters.n_players)],
                [states[position_indices[player, 2], 1:i] for player in 1:(parameters.n_players)],
                linewidth = line_width,
            )
        end
        # plot goals from parameters info with an x
        Plots.scatter!(
            [parameters.goals[player][1] for player in 1:(parameters.n_players)],
            [parameters.goals[player][2] for player in 1:(parameters.n_players)],
            markersize = marker_size_goals,
            marker = :star4,
            color = colors,
        )

        # Plot KoZs
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
                )
            end
            # Plot hyperplane normal
            ni =
                parameters.ρs[couple_idx] *
                n(i, θs[couple_idx], parameters.αs[couple_idx], parameters.ωs[couple_idx])
            # Plots.plot!(
            #     [states[position_indices[couple[2], 1], i], states[position_indices[couple[2], 1], i] + ni[1]],
            #     [states[position_indices[couple[2], 2], i], states[position_indices[couple[2], 2], i] + ni[2]],
            #     arrow = true,
            #     color = colors[couple[1]],
            # )
            # Plot hyperplane 
            hyperplane_domain = 10*range(domain[1],domain[2],100)
            p = states[position_indices[couple[2], 1:2], i] + ni
            Plots.plot!(hyperplane_domain .+ p[1],
                [-ni[1]/ni[2]*x + p[2] for x in hyperplane_domain],
                color = colors[couple[1]],
                linewidth = line_width_hyperplanes,
                linestyle = :dot,
            )
        end

        # Plot player positions on top 
        Plots.scatter!(
            [states[position_indices[player, 1], i] for player in 1:(parameters.n_players)],
            [states[position_indices[player, 2], i] for player in 1:(parameters.n_players)],
            markersize = marker_size_players,
            color = colors,
        )

        # Set domain
        Plots.plot!(xlims = domain,
              ylims = domain)

        # Save if at saveframe
        if !isnothing(save_frame) && i == save_frame
            Plots.savefig("figures/hyperplanes_frame_"*filename*".png")
        end
    end
    Plots.gif(anim, fps = fps, "figures/hyperplanes_"*filename*".gif")
end    

function main()
    # Game parameters
    horizon = 44
    dt = 5.0
    initial_state = mortar([[-100.0, 0.0, 0.0, 0.0], [0.0, -100.0, 0.0, 0.0]])
    goals = mortar([[100.0, 0.0], [0.0, 100.0]]) # TEMPORARY 
    ωs = [0.015]
    α0s = [3*pi/4]
    ρs = [30.0]
    couples = [(1,2)]
    weights = mortar([[10.0, 0.0001], [10.0, 0.0001]])
    n_players = 2
    n_states_per_player = 4 
    m   = 100.0 # kg
    r₀ = (400 + 6378.137) # km
    grav_param  = 398600.4418 # km^3/s^2
    n = sqrt(grav_param/(r₀^3)) # rad/s   

    # Parameters 
    θ = mortar([initial_state, goals, weights, ωs, α0s, ρs])

    game = setup_trajectory_game(; dt, n, m, couples)
    parametric_game = build_parametric_game(; game, horizon, N = n_players, n_couples = length(couples))
    
    turn_length = horizon # verify this. 
    sim_steps = let
        n_sim_steps = horizon # verify this
        progress = ProgressMeter.Progress(n_sim_steps)
        receding_horizon_strategy =
            WarmStartRecedingHorizonStrategy(; game, parametric_game, turn_length, horizon, parameters = θ)

        rollout(
            game.dynamics,
            receding_horizon_strategy,
            θ[Block(1)],
            n_sim_steps;
            get_info = (γ, x, t) ->
                (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
        )
    end

    # Save video 
    # animate_sim_steps(game, sim_steps; live = false, framerate = 20, show_turn = true, xlims = (-200, 200), ylims = (-200, 200))

    # Plot hyperplanes
    solution = (; x = hcat(sim_steps.xs...), u = hcat(sim_steps.us...))
    adjacency_matrix = [false true; false false] # used in plotting only
    plot_parameters = 
        (;
            n_players,
            n_states_per_player,
            goals = [goals[Block(player)] for player in 1:2],
            adjacency_matrix, 
            couples = findall(adjacency_matrix),
            ωs,
            αs = zeros(length(couples)),
            ρs,
            ΔT = dt
        )
    visualize_rotating_hyperplanes(
        solution.x,
        plot_parameters;
        # title = string(n_players)*"p",
        koz = true,
        fps = 10.0,
        filename = "forward",
        save_frame = 25,
    )

    (;sim_steps, game)
end
