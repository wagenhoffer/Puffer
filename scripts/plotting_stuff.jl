using Puffer
using Plots

begin
    foil, flow = init_params(; N=50, T=Float64, motion=:make_heave_pitch,
        f=0.25, motion_parameters=[0, deg2rad(5)])
    aoa = rotation(8 * pi / 180)'
    foil._foil = (foil._foil' * aoa')'
    wake = Wake(foil)
    (foil)(flow)
    movie = @animate for i = 1:flow.N*5
        # begin
        A, rhs, edge_body = make_infs(foil)
        setσ!(foil, flow)
        cancel_buffer_Γ!(wake, foil)
        wake_ind = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(wake_ind .* foil.normals, dims=1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs*foil.σs-buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        fg, eg = get_circulations(foil)
        # @assert sum(fg)+sum(eg)+sum(wake.Γ) <1e-15
        #DETERMINE Velocities onto the wake and move it        
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        move_wake!(wake, flow)
        # Kutta condition!        
        @assert (-foil.μs[1] + foil.μs[end] - foil.μ_edge[1]) == 0.0

        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)
        wm = maximum(foil.foil[1, :]')
        win = (wm - 1.2, wm + 1.1)
        win = (wm - 0.1, wm + 0.1)
        # a = plot_current(foil, wake; window=win)
        a = plot_current(foil, wake)
        # plot!(a, [foil.edge[1,2],foil.edge[1,2]+10.],[foil.edge[2,2],foil.edge[2,2]], color=:green)
        a

        release_vortex!(wake, foil)
        (foil)(flow)
    end
    fg, eg = get_circulations(foil)
    @show sum(fg), sum(eg), sum(wake.Γ)

    gif(movie, "wake.gif", fps=60)

end

begin
    foil, flow = init_params(; N=50, T=Float64, motion=:make_heave_pitch,
                             f=0.75, motion_parameters=[-0.1, π / 20])
    aoa = rotation(0 * pi / 180)'
    foil._foil = (foil._foil' * aoa')'
    wake = Wake(foil)
    (foil)(flow)
    movie = @animate for i = 1:flow.N*5
        time_increment!(flow, foil, wake)
        plot_current(foil, wake)
        ## Get pressure on foil or other metrics
    end
    gif(movie, "images/no_time_step.gif", fps=60)
end