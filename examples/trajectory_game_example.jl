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

using InverseHyperplanes
using UnPack: @unpack
using BlockArrays: mortar, Block
using TrajectoryGamesBase: state_dim, control_dim, horizon, num_players
using Makie
using Random: MersenneTwister

"Setup parameters for the forward game, inverse game, and the MC analysis parameters"
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