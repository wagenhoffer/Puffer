include("../src/BemRom.jl")
using SpecialFunctions
using Plots


heave_pitch = deepcopy(defaultDict)
heave_pitch[:Nt] = 150
heave_pitch[:Ncycles] = 4
heave_pitch[:f] = 1.5
heave_pitch[:Uinf] = 2.0
heave_pitch[:kine] = :make_heave_pitch
θ0 = deg2rad(6)
h0 = 0.0
heave_pitch[:motion_parameters] = [h0, θ0]

foil, flow = init_params(;heave_pitch...)
wake = Wake(foil)

begin
    foil, flow = init_params(;heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)
        f = plot_current(foil, wake)
        f
    end
    gif(movie, "handp.gif", fps=60)
end

begin
    foil, flow = init_params(;heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)

    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:flow.Ncycles*flow.N
        wake_ind = time_increment!(flow, foil, wake)
        phi =  get_phi(foil, wake)
        p, old_mus, old_phis = panel_pressure(foil, flow, wake_ind, old_mus, old_phis, phi)
        coeffs[:,i] = get_performance(foil, flow, p)
    
    end
    t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = 50 
    a = plot(t[start:end], coeffs[1,start:end], label="Force"  ,lw = 3, marker=:circle)
    b = plot(t[start:end], coeffs[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
    c = plot(t[start:end], coeffs[3,start:end], label="Thrust" ,lw = 3, marker=:circle)
    d = plot(t[start:end], coeffs[4,start:end], label="Power"  ,lw = 3, marker=:circle)
    plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(1000,1000))
end
