
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
function compose_from_models(models, indices, θ)
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
                θ[block_dict[key]][indices[key]] = value
            else
                θ[block_dict[key]] = value
            end
        end
    end

    θ
end

"""
Solve the inverse game taking gradient steps

Observation is a vector of concatenated states for all players i.e. [x^1, ... , x^n_players], wherex^i = [x^i_1, ..., x^i_T], and T is the horizon
"""
function inverse(observation, θ_guess_0, game_setup; max_grad_steps = 10, tol = 1e-1, verbose = false)

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
    model_x0 = (x0 = copy(θ_guess_0[Block(1)])[indices[:x0]],)
    model_v0 = (v0 = copy(θ_guess_0[Block(1)])[indices[:v0]],)
    model_ω = (ω = copy(θ_guess_0[Block(4)]),)
    model_ρ = (ρ = copy(θ_guess_0[Block(6)]),)

    # Setup chain 
    state_tree_x0 = Optimisers.setup(Optimisers.Adam(learning_rate_x0_pos, (0.9, 0.999)), model_x0)
    state_tree_v0 = Optimisers.setup(Optimisers.Adam(learning_rate_x0_vel, (0.8, 0.999)), model_v0)
    state_tree_ω = Optimisers.setup(Optimisers.Adam(learning_rate_ω, (0.8, 0.999)), model_ω)
    state_tree_ρ = Optimisers.setup(Optimisers.Adam(learning_rate_ρ, (0.8, 0.999)), model_ρ)

    # Print first step 
    z_new, status_first, loss_first = inverse_loss(observation, θ_guess_0, game, parametric_game)

    # Print hyperplane parameters horizontally
    verbose && println(
        "0: ",
        "Δx0: ",
        norm(θ_guess_0[Block(1)][indices[:x0]] - θ_truth[Block(1)][indices[:x0]]),
        ", Δv0: ",
        norm(θ_guess_0[Block(1)][indices[:v0]] - θ_truth[Block(1)][indices[:v0]]),
        ", ωs: ",
        trunc.(θ_guess_0[Block(4)], digits = 3),
        ", ρs: ",
        trunc.(θ_guess_0[Block(6)], digits = 3),
        " L:  ",
        loss_first,
    )

    # Break if first step did not converge
    if status_first != PATHSolver.MCP_Solved
        verbose && println( "   Stopping: First step did not converge.") 
        return false, θ_guess_0
    end

    # Gradient wrt hyperplane parameters only 
    grad_norms  = []
    grad_norms_x0 = Float64[]
    grad_norms_v0 = Float64[]
    grad_norms_ω = Float64[]
    grad_norms_ρ = Float64[]
    losses = []
    θ_guess = deepcopy(θ_guess_0)
    for i in 1:max_grad_steps

        # Gradient 
        grad = Zygote.gradient(
            θ -> inverse_loss(observation, θ, game, parametric_game; initial_guess = z_new)[3],
            collect(θ_guess),
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
