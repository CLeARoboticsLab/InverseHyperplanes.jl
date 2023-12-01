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