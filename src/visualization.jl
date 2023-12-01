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