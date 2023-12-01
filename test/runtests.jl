using InverseHyperplanes
using Test: @testset, @test
using BlockArrays: Block, mortar
using Random: MersenneTwister

@testset "OptimizationTests" begin
    f(x, θ) = sum(x)
    g(x, θ) = [sum(x .^ 2) - 1]
    h(x, θ) = -x

    problem = ParametricOptimizationProblem(;
        objective = f,
        equality_constraint = g,
        inequality_constraint = h,
        parameter_dimension = 1,
        primal_dimension = 2,
        equality_dimension = 1,
        inequality_dimension = 2,
    )

    (; primals, variables, status, info) = solve(problem, [0])
    @test all(isapprox.(primals, -0.5sqrt(2), atol = 1e-6))
end

@testset "GameTests" begin
    N = 3
    fs = [(x, θ) -> sum(x[Block(ii)]) for ii in 1:N]
    gs = [(x, θ) -> [sum(x[Block(ii)] .^ 2) - 1] for ii in 1:N]
    hs = [(x, θ) -> -x[Block(ii)] for ii in 1:N]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]

    problem = ParametricGame(
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = 1,
        primal_dimensions = fill(2, N),
        equality_dimensions = fill(1, N),
        inequality_dimensions = fill(2, N),
        shared_equality_dimension = 1,
        shared_inequality_dimension = 1,
    )

    (; primals, variables, status, info) = solve(problem, [0])
    for x in primals
        @test all(isapprox.(x, -0.5sqrt(2), atol = 1e-6))
    end
end

@testset "Ground-truth learning" begin
    # Setup game with known parameters
    function setup_experiment(;n_players = 2)

        function unitvector(θ)
            [cos(θ), sin(θ)]
        end
    
        # ---- FORWARD GAME PARAMETERS ----
    
        # Game parameters
        horizon = 15
        dt = 10.0
        scale = 100.0
        initial_state = mortar([vcat(-scale .* unitvector(pi/n_players*(i-1)), [0.0,0.0]) for i in 1:n_players])
        goals = mortar([scale .* unitvector(pi/n_players*(i-1)) for i in 1:n_players])
        couples = [(i, j) for i in 1:n_players for j in i+1:n_players]
        ωs = [0.015 for _ in couples]
        α0s = [
            atan(
                initial_state[Block(couple[1])][2] - initial_state[Block(couple[2])][2],
                initial_state[Block(couple[1])][1] - initial_state[Block(couple[2])][1],
            ) for couple in couples
        ]
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
        learning_rate_x0_pos = 0.0
        learning_rate_x0_vel = 0.0
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

    experiment_setup = setup_experiment()

    # Compute forward trajectory and extract player states
    solution_forward = forward(experiment_setup.θ_truth, experiment_setup; visualize = false)
    primals_forward = solution_forward.variables[1:sum(experiment_setup.parametric_game.primal_dimensions)]
    states_forward = extract_states(primals_forward, experiment_setup.game.dynamics)

    # Learn parameters by solving inverse game
    converged_inverse, inverse_parameters = inverse(
        states_forward,
        experiment_setup.θ_guess,
        experiment_setup;
        max_grad_steps = 100,
        verbose = false,
    )

    # Test
    @test converged_inverse
    @test isapprox(inverse_parameters[Block(4)], experiment_setup.θ_truth[Block(4)], atol = 1e-2) #ω
    @test isapprox(inverse_parameters[Block(6)], experiment_setup.θ_truth[Block(6)], atol = 5) #ρ
end
