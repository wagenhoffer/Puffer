include("../src/BemRom.jl")
using SpecialFunctions
using Plots


fixedangle = deepcopy(defaultDict)
fixedangle[:N] = 50
fixedangle[:Nt] = 64
fixedangle[:Ncycles] = 2
fixedangle[:f] = 0.25
fixedangle[:Uinf] = 1
fixedangle[:kine] = :make_heave_pitch
θ0 = deg2rad(10)
h0 = 0.0
fixedangle[:motion_parameters] = [h0, θ0]
fixedangle[:aoa] = deg2rad(33)

# delete!(fixedangle, :motion_parameters)

begin
    foil, flow = init_params(;fixedangle...)
    foil._foil = (foil._foil' * rotation(-fixedangle[:aoa])')'
    wake = Wake(foil)
    (foil)(flow)
    #LESP
    set_ledge!(foil, flow)
    #data containers
    old_mus, old_phis = zeros(3,foil.N), zeros(3,foil.N)   
    phi = zeros(foil.N)
    coeffs = zeros(4,flow.Ncycles*flow.N)
    ps = zeros(foil.N ,flow.Ncycles*flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    movie = @animate for i in 1:flow.Ncycles*flow.N
        A, rhs, edge_body = make_infs(foil)
        #LESP
        if i%1 == 0
            le_inf, le_buff = ledge_inf(foil)
            A = A + le_inf
        end        
        setσ!(foil, flow)    
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        #LESP
        if i%1 == 0 
            buff += le_buff * foil.μ_ledge[1]
        end
        foil.μs = A \ (-rhs*foil.σs-buff)[:]
        set_edge_strength!(foil)
        #LESP
        if i%1 == 0  
            set_ledge_strength!(foil)
        else
            foil.μ_ledge[2] = foil.μ_ledge[1]
            foil.μ_ledge[1] = 0.0
        end
        @show foil.μ_ledge
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)  
        phi =  get_phi(foil, wake)                                   
        # p, old_mus, old_phis = panel_pressure(foil, flow,  old_mus, old_phis, phi)        
        # coeffs[:,i] = get_performance(foil, flow, p)
        # ps[:,i] = p

        move_wake!(wake, flow)
        release_vortex!(wake, foil)
        (foil)(flow)
        #LEading edge is separate for now, rollin it into the main loop        
        #LESP
        set_ledge!(foil, flow)
        f = plot_current(foil, wake)
        f
    end
    gif(movie, "LESP_1.gif", fps=30)  
end


foil, flow = init_params(;fixedangle...)
wake = Wake(foil)

begin
    foil, flow = init_params(;fixedangle...)
    if haskey(fixedangle, :aoa)
        foil._foil = (foil._foil' * rotation(-fixedangle[:aoa])')'            
    end
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOPwake.uv = [wake.uv .* 0.0 [0.0, 0.0]]
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
    foil, flow = init_params(;fixedangle...)
    foil._foil = (foil._foil' * rotation(-fixedangle[:aoa])')'
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
        p, old_mus, old_phis = panel_pressure(foil, flow,  old_mus, old_phis, phi)        
        coeffs[:,i] = get_performance(foil, flow, p)
        ps[:,i] = p
    end
    mid = foil.N ÷ 2 +1
    plot(foil.col[1,:], ps[:,end], label="P")
    plot!(foil.col[1,mid-2:mid+2], ps[mid-2:mid+2,end], marker=:circle, label="LESP", legend=:topleft,lw=0)
end


function plot_ledge(foil::Foil, ledge)
    a = plot(foil.foil[1,:],foil.foil[2,:], label="Foil")
    plot!(a, foil.col[1,:], foil.col[2,:], marker=:dot, lw = 0, label="")
    plot!(a, ledge[1,:], ledge[2,:], label="LESP")
    a
end

function set_ledge!(foil::Foil, flow::FlowParams)
    front = foil.N ÷ 2 +1
    le_norm = foil.normals[:,front+1]*flow.Uinf*flow.Δt
    #initial the ledge
    if flow.n ==1
        foil.ledge = [foil.foil[:,front] foil.foil[:,front] .+ le_norm*0.4 foil.foil[:,front] .+ le_norm*1.4 ]
    else
        foil.ledge = [foil.foil[:,front] foil.foil[:,front] .+ le_norm*0.4 foil.ledge[:,2]]  
    end
end





function ledge_inf(foil::Foil)
    x1, x2, y = panel_frame(foil.col, foil.ledge)
    edgeInf = doublet_inf.(x1, x2, y)
    edgeMat = zeros(foil.N,foil.N)
    #TODO: make this the correct spot for sure
    mid = foil.N ÷ 2
    edgeMat[:, mid] = edgeInf[:, 1]
    edgeMat[:, mid+1] = -edgeInf[:, 1]
    edgeMat, edgeInf[:,2]
end