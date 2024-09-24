# include("../src/BemRom.jl")
using Puffer
using BSplineKit
using Plots
###functions to move to BemRom###
function plot_ledge(foil::Foil)
    a = plot(foil.foil[1, :], foil.foil[2, :],  label = "Foil")    
    plot!(a, foil.ledge[1, :], foil.ledge[2, :], label = "LESP", aspect_ratio = :equal)
    a
end

function set_ledge!(foil::Foil, flow::FlowParams)
    front = foil.N ÷ 2
    le_norm = sum(foil.normals[:, front:(front + 1)], dims = 2) / 2.0 * flow.Uinf * flow.Δt
    #initial the ledge
    front += 1
    foil.ledge = [foil.foil[:,front] foil.foil[:,front] .+ le_norm*0.4 foil.foil[:,front] .+ le_norm*1.4 ]
    # if flow.n ==1
    #     foil.ledge = [foil.foil[:,front] foil.foil[:,front] .+ le_norm*0.4 foil.foil[:,front] .+ le_norm*1.4 ]
    # else
    #     foil.ledge = [foil.foil[:,front] foil.foil[:,front] .+ le_norm*0.4 foil.ledge[:,2]]  
    # end
end

function ledge_inf(foil::Foil)
    x1, x2, y = panel_frame(foil.col, foil.ledge)
    edgeInf = doublet_inf.(x1, x2, y)
    edgeMat = zeros(foil.N, foil.N)
    #TODO: make this the correct spot for sure
    mid = foil.N ÷ 2
    edgeMat[:, mid] = edgeInf[:, 1]
    edgeMat[:, mid + 1] = -edgeInf[:, 1]
    edgeMat, edgeInf[:, 2]
end

function get_μ!(foil::Foil, rhs, A, le_inf, buff, lesp)

    #now pseudo code, but if lesp, then add le_inf to A, else the simulation was fine
    if !lesp
        foil.μs = A \ (-rhs * foil.σs - buff)[:]
    else
        foil.μs = (A + le_inf) \ (-rhs * foil.σs - buff)[:]       
    end    
    nothing
end
### ###

fixedangle = deepcopy(defaultDict)
fixedangle[:N] = 64
fixedangle[:Nt] = 64
fixedangle[:Ncycles] = 3
fixedangle[:f] = 0.5
fixedangle[:Uinf] = 1.0
# fixedangle[:kine] = :make_eldredge
fixedangle[:kine] = :make_heave_pitch
fixedangle[:thick] = 0.12
θ0 = deg2rad(30.0)
h0 = 0.0
fixedangle[:motion_parameters] = [h0, θ0]
# fixedangle[:motion_parameters] = [θ0, 1000.]
fixedangle[:aoa] = deg2rad(0.0)
fixedangle[:pivot] = 0.0

# delete!(fixedangle, :motion_parameters)
global lesp = false
global old_phis = zeros(3, fixedangle[:N])
global old_mus = zeros(3, fixedangle[:N])
begin
    foil, flow = init_params(; fixedangle...)
    foil._foil = (foil._foil' * rotation(-fixedangle[:aoa])')'
    wake = Wake(foil)
    (foil)(flow)
    #LESP
    set_ledge!(foil, flow)
    #data containers
    old_phis = zeros(3, foil.N)
    old_mus = zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)
    pus = zeros(foil.N, flow.Ncycles * flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    animated = true
    anim = Animation()
    # movie = @animate for i in 1:flow.Ncycles*flow.N
    front = foil.N ÷ 2
    dest = [0.0 0.0]'
    lespcont(p) = true #length(filter(x -> x > 1e8,  p[(front - 1):(front + 1)])) > 0
    for i in 1:Int((flow.Ncycles * flow.N)*.5 )
        if i != 1                        
            
            # dest = hcat(wake.xy[:,1:2], dest)
            wake.xy = sdf_fence(wake, foil, flow;dest = dest)
            release_vortex!(wake, foil)
        end
        (foil)(flow)
        set_ledge!(foil, flow)
        A, rhs, edge_body = make_infs(foil)
        le_inf, le_buff = ledge_inf(foil)
        setσ!(foil, flow)
        #doesn't change
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        #doesn't change
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims = 1)'
        # Repeated 
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        buff += le_buff * foil.μ_ledge[1]
        lesp = false
        get_μ!(foil, rhs, A, le_inf, buff, lesp)
        p = panel_pressure(foil, flow, old_mus, old_phis, phi)
        if lespcont(p)
            lesp = true
            get_μ!(foil, rhs, A, le_inf, buff, lesp)                        
            p = panel_pressure(foil, flow, old_mus, old_phis, phi)
        else
            @show "no LESP"            
            foil.μ_ledge[2] = foil.μ_ledge[1]
            foil.μ_ledge[1] = 0.0
        end  
        set_ledge_strength!(foil; lesp = lesp)
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        
        coeffs[:, i] = get_performance(foil, flow, p)        

        body_to_wake!(wake, foil, flow)
        #ledge to wake velocity
        # wake.uv .+= vortex_to_target(foil.ledge, wake.xy, [-foil.μ_ledge[1], foil.μ_ledge[1]-foil.μ_ledge[2], foil.μ_ledge[2] ], flow)
        wake_self_vel!(wake, flow)
        dest = wake.xy + wake.uv * flow.Δt
        old_mus = [foil.μs'; old_mus[1:2, :]]
        old_phis = [phi'; old_phis[1:2, :]]
 
        if animated        
            f = plot(foil, wake)
            frame(anim, f)
        end
        ps[:, i] = p
    end
    if animated
        gif(anim, "LESP_1.gif", fps = 30)
    end
end
begin
    acc_lens = 0.5 * foil.panel_lengths[1:(end - 1)] + 0.5 * foil.panel_lengths[2:end]
    acc_lens = [0; cumsum(acc_lens)]

    S = interpolate(acc_lens, p[:], BSplineOrder(3))
    dp = Derivative(1) * S
    dp = -dmu.(acc_lens)
    plot(dp)
end

begin
    t = LinRange(0, flow.Ncycles / foil.f, flow.Ncycles * flow.N)
    a1 = contour(t, 1:(foil.N), (ps + pus), colors = :jet, label = "")
    angle = foil.kine[2].(0.0, t, 0.0) * 180.0 / pi

    b2 = plot(t, angle, size = (800, 200), label = "")
    plot(a1, b2, layout = (2, 1), size = (800, 800))
end

foil, flow = init_params(; fixedangle...)
wake = Wake(foil)

begin
    foil, flow = init_params(; fixedangle...)
    if haskey(fixedangle, :aoa)
        foil._foil = (foil._foil' * rotation(-fixedangle[:aoa])')'
    end
    wake = Wake(foil)
    (foil)(flow)
    ### EXAMPLE OF AN ANIMATION LOOPwake.uv = [wake.uv .* 0.0 [0.0, 0.0]]
    movie = @animate for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        # Nice steady window for plotting
        win = (minimum(foil.foil[1, :]') - foil.chord / 2.0,
            maximum(foil.foil[1, :]) + foil.chord * 2)
        #ZOom in on the edge
        # win = (minimum(foil.foil[1, :]') + 3*foil.chord / 4.0, maximum(foil.foil[1, :]) + foil.chord * 0.1)
        win = nothing
        f = plot_current(foil, wake; window = win)
        f
    end
    gif(movie, "handp.gif", fps = 10)
end
begin
    f = plot_current(foil, wake)
    f
end
begin
    foil, flow = init_params(; fixedangle...)
    foil._foil = (foil._foil' * rotation(-fixedangle[:aoa])')'
    k = foil.f * foil.chord / flow.Uinf
    @show k
    wake = Wake(foil)
    (foil)(flow)
    #data containers
    old_mus, old_phis = zeros(3, foil.N), zeros(3, foil.N)
    phi = zeros(foil.N)
    coeffs = zeros(4, flow.Ncycles * flow.N)
    ps = zeros(foil.N, flow.Ncycles * flow.N)
    ### EXAMPLE OF AN PERFROMANCE METRICS LOOP
    for i in 1:(flow.Ncycles * flow.N)
        time_increment!(flow, foil, wake)
        phi = get_phi(foil, wake)
        p, old_mus, old_phis = panel_pressure(foil, flow, old_mus, old_phis, phi)
        coeffs[:, i] = get_performance(foil, flow, p)
        ps[:, i] = p
    end
    mid = foil.N ÷ 2 + 1
    plot(foil.col[1, :], ps[:, end], label = "P")
    plot!(foil.col[1, (mid - 2):(mid + 2)],
        ps[(mid - 2):(mid + 2), end],
        marker = :circle,
        label = "LESP",
        legend = :topleft,
        lw = 0)
end

function plot_ledge(foil::Foil, ledge)
    a = plot(foil.foil[1, :], foil.foil[2, :], label = "Foil")
    plot!(a, foil.col[1, :], foil.col[2, :], marker = :dot, lw = 0, label = "")
    plot!(a, ledge[1, :], ledge[2, :], label = "LESP")
    a
end

function ledge_inf(foil::Foil)
    x1, x2, y = panel_frame(foil.col, foil.ledge)
    edgeInf = doublet_inf.(x1, x2, y)
    edgeMat = zeros(foil.N, foil.N)
    #TODO: make this the correct spot for sure
    mid = foil.N ÷ 2
    edgeMat[:, mid] = edgeInf[:, 1]
    edgeMat[:, mid + 1] = -edgeInf[:, 1]
    edgeMat, edgeInf[:, 2]
end

function get_μ!(foil::Foil, rhs, A, le_inf, buff, lesp)

    #now pseudo code, but if lesp, then add le_inf to A, else the simulation was fine
    if !lesp
        foil.μs = A \ (-rhs * foil.σs - buff)[:]
    else
        foil.μs = (A + le_inf) \ (-rhs * foil.σs - buff)[:]
        set_ledge_strength!(foil)
    end
    set_edge_strength!(foil)

    nothing
end
