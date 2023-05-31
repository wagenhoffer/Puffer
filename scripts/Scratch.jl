include("../src/BemRom.jl")
using SpecialFunctions
using Plots


heave_pitch = deepcopy(defaultDict)
heave_pitch[:Nt] = 64
heave_pitch[:Ncycles] = 4
heave_pitch[:f] = 0.25
heave_pitch[:Uinf] = 1
heave_pitch[:kine] = :make_heave_pitch
θ0 = deg2rad(30)
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
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0, maximum(foil.foil[1, :]) + foil.chord * 2)
        #ZOom in on the edge
        # win = (minimum(foil.foil[1, :]') + 3*foil.chord / 4.0, maximum(foil.foil[1, :]) + foil.chord * 0.1)
        win=nothing
        f = plot_current(foil, wake;window=win)
        f
    end
    gif(movie, "handp.gif", fps=10)
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
    plot(a,b,c,d, layout=(2,2), legend=:topleft, size =(1000,1000))
end

begin
    # plot a few snap shot at opposite ends of the motion
    pos = 15
    # plot(phis[pos ] .- maximum(phis[pos]), label="x-start")
    # plot!(phis[pos+flow.N÷2][end:-1:1] .- maximum(phis[pos+flow.N÷2]), label="x-half",marker=:circle,lw=0)
    xx = plot(phis[pos][1,:])
    plot!(xx, phis[pos+flow.N÷2][1,end:-1:1])
    yy = plot(phis[pos][2,:])
    plot!(yy, -phis[pos+flow.N÷2][2,end:-1:1])
    plot(xx,yy, layout=(1,2),size=(1000,800))
end
begin
    #look at the panel pressure for a foil
    a = plot()
    pressures = @animate for i = 10:flow.N*flow.Ncycles
        plot(a, foil.col[1,:], ps[:,i]/(0.5*flow.Uinf^2), label="",ylims=(-1,1))
    end
    gif(pressures, "pressures.gif", fps=30)
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
    vx = plot()
    vy = plot()
    for i in 1:flow.N
        (foil)(flow)
        get_panel_vels!(foil, flow)
        plot!(vx, foil.panel_vel[1,:], label="")    
        plot!(vy, foil.panel_vel[2,:], label="")        
    end
    plot(vx,vy, layout=(1,2))
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