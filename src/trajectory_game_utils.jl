
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


"""
Utility to extract states from the primals of an MCP solution.

Primals should be in the form of a single vector. 

Output is the vcat of every player's trajectory [x^1, x^2, ...] where x^i is player i's state trajectory.
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

