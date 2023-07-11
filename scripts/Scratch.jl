include("../src/BemRom.jl")

using SpecialFunctions
using Plots


heave_pitch = deepcopy(defaultDict)
heave_pitch[:N] =12
heave_pitch[:Nt] = 64
heave_pitch[:Ncycles] = 5
heave_pitch[:f] = 1.
heave_pitch[:Uinf] = 2
heave_pitch[:kine] = :make_heave_pitch
θ0 = deg2rad(2)
h0 = 0.1
heave_pitch[:motion_parameters] = [h0, θ0]

foil, flow = init_params(;heave_pitch...)
wake = Wake(foil)

begin
    foil, flow = init_params(;heave_pitch...)
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOP
    movie = @animate for i in 1:flow.Ncycles*flow.N*1.75
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)
        #ZOom in on the edge
        # win = (minimum(foil.foil[1, :]') + 3*foil.chord / 4.0, maximum(foil.foil[1, :]) + foil.chord * 0.1)
        win=nothing
        f = plot_current(foil, wake;window=win)
        plot!(f, ylims=(-1,1))
        plot!(title="$(foil.panel_vel[2,6])")
    end
    gif(movie, "handp.gif", fps=10)
end

begin
    f = plot_current(foil, wake)
    f
    plot!(f, foil.foil[1,:],foil.foil[2,:], marker=:bar,label="Panels")
    plot!(f, cb=:none)
    plot!(f, showaxis=false)
    plot!(f, size=(600,200))
    plot!(f, grid=:off)
end

begin
    foil, flow = init_params(;heave_pitch...)
    k = foil.f*foil.chord/flow.Uinf
    @show k
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:flow.Ncycles*flow.N
        time_increment!(flow, foil, wake)
        phi =  get_phi(foil, wake)                                   
        p = panel_pressure(foil, flow,  old_mus, old_phis, phi)        
        old_mus = [foil.μs'; old_mus[1:2,:]]
        old_phis = [phi'; old_phis[1:2,:]]
        coeffs[:,i] = get_performance(foil, flow, p)
        ps[:,i] = p
    end
    t = range(0, stop=flow.Ncycles*flow.N*flow.Δt, length=flow.Ncycles*flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs[1,start:end], label="Force"  ,lw = 3, marker=:circle)
    b = plot(t[start:end], coeffs[2,start:end], label="Lift"   ,lw = 3, marker=:circle)
    c = plot(t[start:end], coeffs[3,start:end], label="Thrust" ,lw = 3, marker=:circle)
    d = plot(t[start:end], coeffs[4,start:end], label="Power"  ,lw = 3, marker=:circle)
    plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(800,800))
end

begin
    #look at the panel pressure for a foil
    a = plot()
    pressures = @animate for i = 10:flow.N*flow.Ncycles
        plot(a, foil.col[1,:], ps[:,i]/(0.5*flow.Uinf^2), label="",ylims=(-2,5))
    end
    gif(pressures, "../images/pressures.gif", fps=30)
end
begin
    # plot a few snap shot at opposite ends of the motion
    pos = 60
    plot( foil.col[1,:], ps[:,pos]/(0.5*flow.Uinf^2), label="start",ylims=(-2,2))
    plot!( foil.col[1,:], ps[:,pos+flow.N÷2]/(0.5*flow.Uinf^2), label="half",ylims=(-2,2))
end


begin
    #look at the panel velocity for a heaving foil, strictly y-comp
    θ0 = deg2rad(0)
    h0 = 0.1
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(;heave_pitch...)
    t = flow.Δt:flow.Δt:flow.N*flow.Ncycles*flow.Δt
    vx = zeros(flow.N*flow.Ncycles)
    vy = zeros(flow.N*flow.Ncycles)
    for i in 1:flow.N*flow.Ncycles
        (foil)(flow)
        get_panel_vels!(foil, flow)
        vx[i] = foil.panel_vel[1,1]    
        vy[i] = foil.panel_vel[2,1]
    end
    plot(t, vx, label="Vx")
    plot!(t, vy, label="Vy")
    vya = zeros(size(t))
    @. vya = 2π*foil.f*h0*cos(2π*foil.f*t)
    plot!(t, vya, label="Vya")

    @show sum(abs2, vya-vy)
end
begin
    #look at the panel velocity for a pitching foil
    θ0 = deg2rad(5)
    h0 = 0.0
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(;heave_pitch...)
    vx = plot()
    vxs = []
    vy = plot()
    vys = []
    for i in 1:flow.N
        (foil)(flow)
        get_panel_vels!(foil, flow)
        plot!(vx, foil.col[1,:], foil.panel_vel[1,:], label="")    
        plot!(vy, foil.col[1,:], foil.panel_vel[2,:], label="")        
        push!(vxs, foil.panel_vel[1,:])
        push!(vys, foil.panel_vel[2,:])             
             
    end
    plot(vx,vy, layout=(1,2))
    
end


begin
    #look at the panel normals for a pitching foil
    θ0 = deg2rad(5)
    h0 = 0.0
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(;heave_pitch...)
    normies = plot()
    
    normals = @animate for i in 1:flow.N
        (foil)(flow)
    
        f = quiver(foil.col[1,:],foil.col[2,:], 
               quiver=(foil.normals[1,:],foil.normals[2,:]), label="")    
        f
    end
    gif(normals, "normals.gif", fps=30)
end
begin
    #look at the panel tangents for a pitching foil
    θ0 = deg2rad(5)
    h0 = 0.0
    heave_pitch[:motion_parameters] = [h0, θ0]
    foil, flow = init_params(;heave_pitch...)
    normies = plot()
    
    normals = @animate for i in 1:flow.N
        (foil)(flow)
    
        f = quiver(foil.col[1,:],foil.col[2,:], 
               quiver=(foil.tangents[1,:],foil.tangents[2,:]), label="")    
        f
    end
    gif(normals, "tangents.gif", fps=30)
end


p