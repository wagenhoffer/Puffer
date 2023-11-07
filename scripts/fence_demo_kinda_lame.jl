# include("../src/BemRom.jl")
using Puffer

using Plots
using Dierckx


begin
    heave_pitch = deepcopy(defaultDict)
    heave_pitch[:N] = 64
    heave_pitch[:Nt] = 64
    heave_pitch[:Ncycles] = 1
    heave_pitch[:f] = 2.0
    heave_pitch[:Uinf] = 1.0
    heave_pitch[:kine] = :make_heave_pitch
    θ0 = deg2rad(1)
    h0 = 0.01
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(; heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)

    for i in 1:(flow.Ncycles * flow.N * 2)
        time_increment!(flow, foil, wake)
        dest = wake.xy + wake.uv .* flow.Δt
    end
    #We only care if the vortex ends up inside of the foil after motion
    nfoil = next_foil_pos(foil, flow)
    a = plot_current(foil, wake)
    wake.xy .-= [2, 0.0]
    a = plot_current(foil, wake)
end

begin
    @show flow.n
    movie = @animate for i in 1:(flow.N * 3)
        dest = sdf_fence!(wake, foil, flow)
        wake.xy = dest
        time_increment!(flow, foil, wake)
        a = plot_current(foil, wake)
        a
    end
    gif(movie, "fence.gif")
end

begin
    plot(nfoil[1, :], top.(nfoil[1, :]))
    plot!(nfoil[1, :], bottom.(nfoil[1, :]))
    plot!(foil.foil[1, :], topcurr.(foil.foil[1, :]))
    plot!(foil.foil[1, :], bottomcurr.(foil.foil[1, :]))
    @show sdf(dest[:, 13])
    plot!([dest[1, 13]], [dest[2, 13]], st = :scatter)
end


