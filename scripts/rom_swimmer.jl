########################################################################
"""
Reduced order model swimmer
requirements: 
1. vortex based
2. encodes transient traveling wave motion
3. includes a pressure solver -> finds efficiency, thrust, power, lift, etc …
    a. Neural Net pressure solver, the remainder of the system is linear per time step and
       this can be avoided and sped up sig if done right
4. easy to setup discrete motions -> faster, slower, left, right
5. invertable? or just attach a FMM code for acceleration
6. Write a version of Tianjuan's BEM to scrape inviscid data from 
7. POD/PCA or DMD on the vortex distro to define a rom
7a. use CNN to define approx the form-> look into latent space and invert for ROM
8. ROM -> P(NN) -> Hydrodynamic performance metrics  
9. Emit a vortex at reg interval

P(Γ((x,y)(t))) we want to reduce Γ while still getting pressure along the body

"""

using LinearAlgebra
using Plots
using BenchmarkTools
using StaticArrays
using ForwardDiff
using BSplineKit

""" data types """
abstract type Body end

mutable struct FlowParams{T}
    Δt::T #time-step
    Uinf::T
    ρ::T #fluid density
    N::Int #number of time steps
    n::Int #current time step
    δ::T #desing radius
end
mutable struct Foil{T} <: Body
    kine  #kinematics: heave or traveling wave function
    f::T #wave freq
    k::T # wave number
    N::Int # number of elements
    _foil::Matrix{T} #coordinates in the Body Frame
    foil::Matrix{T}  # Absolute fram
    col::Matrix{T}  # collocation points
    σs::Vector{T} #source strengths on the body 
    μs::Vector{T} #doublet strengths on the body
    edge::Matrix{T}
    μ_edge::Vector{T}
    chord::T
    normals::Matrix{T}
    tangents::Matrix{T}
    panel_lengths::Vector{T}
    panel_vel::Matrix{T}
end

mutable struct Wake{T}
    xy::Matrix{T}
    Γ::Vector{T}
    uv::Matrix{T}
end
function Base.show(io::IO,w::Wake) 
    print(io,"Wake x,y = ($(w.xy[1,:])\n")
    print(io,"            $(w.xy[2,:]))\n")
    print(io,"     Γ   = ($(w.Γ))")
end
""" functions """
Wake() = Wake([0, 0], 0.0, [0, 0])
#initial with the buffer panel to be cancelled
function Wake(foil::Foil{T}) where {T<:Real}
    Wake{T}(reshape(foil.edge[:, end], (2, 1)), [-foil.μ_edge[end]], [0.0 0.0]')
end

# NACA0012 Foil
function make_naca(N; chord=1, thick=0.12)
    # N = 7
    an = [0.2969, -0.126, -0.3516, 0.2843, -0.1036]
    # T = 0.12 #thickness
    yt(x_) = thick / 0.2 * (an[1] * x_^0.5 + an[2] * x_ + an[3] * x_^2 + an[4] * x_^3 + an[5] * x_^4)
    #neutral x
    x = (1 .- cos.(LinRange(0, pi, (N + 2) ÷ 2))) / 2.0
    foil = [[x[end:-1:1]; x[2:end]]'
            [-yt.(x[end:-1:1]); yt.(x[2:end])]']
    foil .* chord
end
function make_waveform(a0=0.1, a=[0.367, 0.323, 0.310]; T=Float64)
    a0 = T(a0)
    a = a .|> T
    f = k = T(1)

    amp(x, a) = a[1] + a[2] * x + a[3] * x^2
    h(x, f, k, t) = a0 * amp(x, a) * sin(2π * (k * x - f * t)) .|> T
    # h(x,t) = f,k -> h(x,f,k,t)
    h
end
function make_ang(a0=0.1, a=[0.367, 0.323, 0.310])
    a0 = a0
    a = a
    f = π
    k = 0.5

    amp(x, a) = a[1] + a[2] * x + a[3] * x^2
    h(x, f, k, t) = a0 * amp(x, a) * sin(2π * (k * x - f * t))
    # h(x,t) = f,k -> h(x,f,k,t)
    h
end
function no_motion(;T=Float64)
    sig(x,f,k,t) = 0.0
end
function angle_of_attack(;aoa = 5, T=Float64)
    sig(x,f,k,t) = rotate(-aoa*pi/180)'
end

function norms(foil)
    dxdy = diff(foil, dims=2)
    lengths = sqrt.(sum(abs2, diff(foil, dims=2), dims=1))
    tx = dxdy[1, :]' ./ lengths
    ty = dxdy[2, :]' ./ lengths
    # tangents x,y normals x, y  lengths
    return [tx; ty], [-ty; tx], lengths
end
function norms!(foil::Foil)
    dxdy = diff(foil.foil, dims=2)
    lengths = sqrt.(sum(abs2, dxdy, dims=1))
    tx = dxdy[1, :]' ./ lengths
    ty = dxdy[2, :]' ./ lengths
    # tangents x,y normals x, y  lengths
    foil.tangents = [tx; ty]
    foil.normals = [-ty; tx]
    foil.panel_lengths = [lengths...]
    nothing
end

source_inf(x1, x2, z) = (x1 * log(x1^2 + z^2) - x2 * log(x2^2 + z^2) - 2 * (x1 - x2)
                         +
                         2 * z * (atan(z, x2) - atan(z, x1))) / (4π)
doublet_inf(x1, x2, z) = -(atan(z, x2) - atan(z, x1)) / (2π)

"""
    init_params(;N=128, chord = 1.0, T=Float32 )

TODO Add wake stuff
"""
function init_params(; N=128, chord=1.0, T=Float32, motion=:make_ang, f =1 , k = 1)
    N = N
    chord = T(chord)
    naca0012 = make_naca(N + 1; chord=chord) .|> T
    
    ang = eval(motion)()
    fp = FlowParams{T}(1.0 / 150.0, 1, 1000.0, 150, 0, 1.0 / 150.0 *1.3)
    txty, nxny, ll = norms(naca0012)
    #TODO: 0.001 is a magic number for now
    col = (get_mdpts(naca0012) .+  repeat(0.001 .*ll,2,1).*-nxny).|> T
    
    edge_vec = fp.Uinf * fp.Δt * [(txty[1, end] - txty[1, 1]) / 2.0, (txty[2, end] - txty[2, 1]) / 2.0] .|> T
    edge = [naca0012[:, end] (naca0012[:, end] .+ edge_vec) (naca0012[:, end] .+ 2 * edge_vec)]
    #   kine, f, k, N, _foil,    foil ,    col,             σs, μs,         edge, chord, normals, tangents, panel_lengths,panbel_vel
    Foil{T}(ang, T(f), T(k), N, naca0012, copy(naca0012), col, zeros(T, N), zeros(T, N), edge, zeros(T, 2), 1, nxny, txty, ll[:], zeros(size(nxny))), fp
end

get_mdpts(foil) = (foil[:, 2:end] + foil[:, 1:end-1]) ./ 2

function move_edge!(foil::Foil, flow::FlowParams)
    edge_vec = flow.Uinf * flow.Δt * [(foil.tangents[1, end] - foil.tangents[1, 1]) / 2.0, (foil.tangents[2, end] - foil.tangents[2, 1]) / 2.0]
    foil.edge = [foil.foil[:, end] (foil.foil[:, end] .+ edge_vec) (foil.foil[:, end] .+ 2 * edge_vec)]
    nothing
end

function set_collocation!(foil::Foil, S = 0.001)
    foil.col = (get_mdpts(foil.foil) .+  repeat(S .*foil.panel_lengths',2,1).*-foil.normals)
end
"""
    (foil::Foil)(fp::FlowParams)

Use a functor to advance the time-step
"""
function (foil::Foil)(flow::FlowParams)
    #perform kinematics
    #TODO: only modifies the y-coordinates
    foil.foil[2, :] = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, flow.n * flow.Δt)
    #Advance the foil in flow
    foil.foil .+= [-flow.Uinf, 0] .* flow.Δt    
    norms!(foil)
    set_collocation!(foil)    
    move_edge!(foil, flow)
    flow.n += 1
end


"""
    get_panel_vels(foil::Foil,fp::FlowParams)

Autodiff the panel velocities, right now it is only moving in the y-dir and x-dir is free-stream
"""
function get_panel_vels(foil::Foil, fp::FlowParams)
    _t = fp.n * fp.Δt
    col = get_mdpts(foil._foil)
    vy = ForwardDiff.derivative(t -> foil.
        kine.(col[1, :], foil.f, foil.k, t), _t)
    # vy = (vy[1:end-1]+vy[2:end])/2.        
    [zeros(size(vy)) vy]
end

function get_panel_vels!(foil::Foil, fp::FlowParams)
    _t = fp.n * fp.Δt
    col = get_mdpts(foil._foil)
    vy = ForwardDiff.derivative(t -> 
        foil.kine.(col[1, :], foil.f, foil.k, t), _t)
    # vy = (vy[1:end-1]+vy[2:end])/2.            
    foil.panel_vel = [zeros(size(vy))'; vy']
    nothing
end
"""
    panel_frame(target,source)

TBW
"""
function panel_frame(target, source)
    _, Ns = size(source)
    _, Nt = size(target)
    Ns -= 1 #TE accomodations
    x1 = zeros(Ns, Nt)
    x2 = zeros(Ns, Nt)
    y = zeros(Ns, Nt)
    ts, _, _ = norms(source)
    txMat = repeat(ts[1, :]', Nt, 1)
    tyMat = repeat(ts[2, :]', Nt, 1)
    dx = repeat(target[1, :], 1, Ns) - repeat(source[1, 1:end-1]', Nt, 1)
    dy = repeat(target[2, :], 1, Ns) - repeat(source[2, 1:end-1]', Nt, 1)
    x1 = dx .* txMat + dy .* tyMat
    y = -dx .* tyMat + dy .* txMat
    x2 = x1 - repeat(sum(diff(source, dims=2) .* ts, dims=1), Nt, 1)
    x1, x2, y
end

function edge_circulation(foil::Foil)
    -foil.μ_edge[1], foil.μ_edge[1] - foil.μ_edge[2], foil.μ_edge[2]
end

function make_infs(foil::Foil; ϵ=1e-10)
    x1, x2, y = panel_frame(foil.col, foil.foil)
    # ymask = abs.(y) .> ϵ
    # y = y .* ymask
    doubletMat = doublet_inf.(x1, x2, y)
    sourceMat = source_inf.(x1, x2, y)
    x1, x2, y = panel_frame(foil.col, foil.edge)
    edgeInf = doublet_inf.(x1, x2, y)
    edgeMat = zeros(size(doubletMat))
    edgeMat[:, 1] = -edgeInf[:, 1]
    edgeMat[:, end] = edgeInf[:, 1]
    A = doubletMat + edgeMat
    # A, sourceMat, edge panel mu
    doubletMat + edgeMat, sourceMat, edgeInf[:,2]
end

""" 
    setσ!(foil::Foil, wake::Wake,flow::FlowParams)

induced velocity from vortex wake
velocity from free stream
velocity from motion 
"""
function setσ!(foil::Foil,  flow::FlowParams;)    
    #no body motion in the x-dir ...yet #TODO
    get_panel_vels!(foil,flow)
    # vy = ForwardDiff.derivative(t -> foil.kine.(foil._foil[1, :], foil.f, foil.k, t), flow.Δt * flow.n)
    # vy = (vy[1:end-1] + vy[2:end]) / 2.0 #averaging, ugh...
    foil.σs = (-flow.Uinf  .+ foil.panel_vel[1,:]) .* foil.normals[1, :] +
                            ( foil.panel_vel[2,:]) .* foil.normals[2, :]
    nothing
end



function move_wake!(wake::Wake, flow::FlowParams)
    # wake.uv = vortex_to_target(wake.xy, wake.xy, wake.xy; δ=flow.δ)
    wake.xy += wake.uv .* flow.Δt
    nothing
end

function wake_self_vel!(wake::Wake, flow::FlowParams)
    wake.uv .+= vortex_to_target(wake.xy, wake.xy, wake.Γ; δ=flow.δ)    
    nothing
end

"""
    body_to_wake!(wake :: Wake, foil :: Foil)

Influence of body onto the wake and the wake onto the wake
"""
function body_to_wake!(wake::Wake, foil::Foil)

    x1, x2, y = panel_frame(wake.xy, foil.foil)
    nw, nb = size(x1)
    lexp = zeros((nw, nb))
    texp = zeros((nw, nb))
    yc = zeros((nw, nb))
    xc = zeros((nw, nb))
    β = atan.(-foil.normals[1, :], foil.normals[2, :])
    β = repeat(β, 1, nw)'
    @. lexp = log((x1^2 + y^2) / (x2^2 + y^2)) / (4π)
    @. texp = (atan(y, x2) - atan(y, x1)) / (2π)
    @. xc = lexp * cos(β) - texp * sin(β)
    @. yc = lexp * sin(β) + texp * cos(β)
    wake.uv = [xc * foil.σs yc * foil.σs]'
    #cirulatory effects    
    #bOUND CIRCULATION (skip kutta for now) Γe2 Γe3(buffer) = μe2
    # Γs = -[diff(foil.μs)... diff(foil.μ_edge)... -foil.μ_edge[end]]
    # ps = hcat(foil.foil[:, 2:end-1], foil.edge[:, 2:end])
    Γs = -[diff(foil.μs)... -foil.μ_edge[end]]
    ps = hcat(foil.foil[:, 2:end-1], foil.edge[:, end])
    # Γs = -[diff(foil.μs)... diff(foil.μ_edge)... ]
    # ps = hcat(foil.foil[:, 2:end-1], foil.edge[:, 2])
    wake.uv .+= vortex_to_target(ps, wake.xy, Γs)
    nothing
end

function vortex_to_target(sources, targets, Γs; δ=0.013)
    ns = size(sources)[2]
    nt = size(targets)[2]
    vels = zeros((2, nt))
    vel = zeros(nt)
    for i = 1:ns
        dx = targets[1, :] .- sources[1, i]
        dy = targets[2, :] .- sources[2, i]
        @. vel = Γs[i] / (2π * (dx^2 + dy^2 + δ^2))
        @. vels[1, :] += dy * vel
        @. vels[2, :] -= dx * vel
    end
    vels
end

function release_vortex!(wake::Wake, foil::Foil)
    # wake.xy = [wake.xy foil.edge[:, end]]
    wake.xy = [wake.xy foil.edge[:, 2]]
    wake.Γ =  [wake.Γ..., (foil.μ_edge[1] - foil.μ_edge[2])]
    wake.uv = [wake.uv [0.0, 0.0]]
    nothing
end

function set_edge_strength!(foil::Foil)
    """Assumes that foil.μs has been set for the current time step 
        TODO: Extend to perform streamline based Kutta condition
    """
    foil.μ_edge[2] = foil.μ_edge[1]
    foil.μ_edge[1] = foil.μs[end] - foil.μs[1]
    nothing
end

function cancel_buffer_Γ!(wake::Wake, foil::Foil)
    #TODO : Add iterator for matching 1->i for nth foil
    wake.xy[:, 1] = foil.edge[:, end]
    wake.Γ[1]     = -foil.μ_edge[end]
    nothing
end

function plot_current(foil::Foil, wake::Wake; window = nothing)
    if !isnothing(window)
        xs = (window[1],window[2])        
    else
        xs=:auto
    end
    a = plot(foil.foil[1, :], foil.foil[2, :], aspect_ratio=:equal, label="")
    plot!(a, foil.edge[1, :], foil.edge[2, :], label="")
    plot!(a, wake.xy[1, :], wake.xy[2, :],
        # markersize=wake.Γ .* 10, st=:scatter, label="",
        markersize=4, st=:scatter, label="",msw=0,xlims=xs,
        marker_z =(wake.Γ), #/mean(abs.(wake.Γ)),
        color=:redsblues)
    a
end

# plot_current(foil,wake; window= (minimum(foil.foil[1,:]')-foil.chord/2.0, maximum(foil.foil[1,:])+foil.chord*5))
# plot_current(foil,wake; window=(-2.2,-1.8))
                                

function plot_with_normals(foil::Foil)
    plot(foil.foil[1, :], foil.foil[2, :], aspect_ratio=:equal, label="")
    quiver!(foil.col[1, :], foil.col[2, :],
        quiver=(foil.normals[1, :], foil.normals[2, :]))
end


function get_qt(foil::Foil)
    acc_lens =  0.5 * foil.panel_lengths[1:end-1]+ 0.5 * foil.panel_lengths[2:end]
    acc_lens = [0;  cumsum(acc_lens)]
    
    # B = BSplineBasis(BSplineOrder(4), acc_lens[:])
    S = interpolate(acc_lens, foil.μs, BSplineOrder(3))    
    dmu = Derivative(1) * S    
    dmudl =  -dmu.(acc_lens)
    
    # plot(dmudl[1:mid],marker=:circle)
    # plot!(-dmudl[end:-1:mid+1], marker=:star)
    
    #TODO: SPLIT σ into local and inducec? it is lumped into one now
    qt = repeat(dmudl',2,1).*foil.tangents #+ repeat(foil.σs',2,1).*foil.normals
    qt
end

#initial allocation - not used (we don't care about the pressures at start now)
function get_dmudt!(foil::Foil, flow::FlowParams)
    old_mus = zeros(3,foil.N)
    old_mus[1,:] = foil.μs
    foil.μs/flow.Δt, old_mus
end
#after init
function get_dmudt!(old_mus, foil::Foil, flow::FlowParams)
    dmudt = (3 * foil.μs - 4 * old_mus[1, :] + old_mus[2, :]) / (2 *flow.Δt)
    old_mus = circshift(old_mus, (1, 0))
    old_mus[1,:] = foil.μs
    dmudt, old_mus
end

rotate(α) = [cos(α) -sin(α)
             sin(α)  cos(α)]
begin
    foil, flow = init_params(; N=50, T=Float64, motion=:no_motion,f=1.1, k=0.55)
    foil._foil = (foil._foil'*aoa')'
    wake = Wake(foil)
    (foil)(flow)
    movie = @animate for i = 1:flow.N
        # begin
        A, rhs, edge_body = make_infs(foil)
        setσ!(foil,  flow)
        wake_ind = vortex_to_target(wake.xy, foil.col, wake.Γ)
        normal_wake_ind = sum(wake_ind .* foil.normals, dims =1 )'
        buff = edge_body * foil.μ_edge[1]
        # B =  -rhs * (foil.σs - normal_wake_ind)  - buff 
        foil.μs = A \ (-rhs * (foil.σs - normal_wake_ind) - buff)[:]      

        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)

        body_to_wake!(wake, foil)
        wake_self_vel!(wake, flow)
        move_wake!(wake, flow)
        # Kutta condition!
        @assert (-foil.μs[1] + foil.μs[end] - foil.μ_edge[1]) == 0.0
        win= (minimum(foil.foil[1,:]')-foil.chord/2.0, maximum(foil.foil[1,:])+foil.chord*5)
        wm = maximum(foil.foil[1,:]')
        win= (wm-1.2, wm+1.1)
        win= (wm- .2, wm+.1)
        a = plot_current(foil, wake; window=win)
        a
        if i!=1 
            release_vortex!(wake, foil)
        end
         #kinematics
        (foil)(flow)
    end
    gif(movie, "wake.gif", fps=60)
end

"""
    run_sim(; steps = flow.N*10, aoa = rotate(-0*pi/180)')


TBW
"""
function run_sim(; steps = flow.N*10, aoa = rotate(-0*pi/180)')    
    # Initialize the foil and flow parameters
    foil, flow = init_params(; N=50, T=Float64, motion=:no_motion)    
    # Rotate the foil based on the angle of attack (aoa)
    foil._foil = (foil._foil' * aoa)'
    # Create a wake object for the foil
    wake = Wake(foil)
    # Initialize the previous time step's vorticity values
    old_mus = zeros(3, foil.N)
    # Initialize arrays to store the pressure coefficients over time
    # cpus = []
    # cps = []
    (foil)(flow)
    for i = 1:steps        
        # # Release vortex from the wake
        # if i != 1
        #     release_vortex!(wake, foil)        
        # end
        # # Update the kinematics of the foil
        
        # Create the influence matrix and right-hand side vector for the foil
        A, rhs, edge_body = make_infs(foil)        
        # Set the surface vorticity of the foil
        setσ!(foil, flow)        
        # Calculate the wake-induced velocity and its normal component
        wake_ind = vortex_to_target(wake.xy, foil.col, wake.Γ)
        normal_wake_ind = sum(wake_ind .* foil.normals, dims=1 )'
        # Calculate the edge vorticity
        buff = edge_body * foil.μ_edge[1]        
        # Calculate the edge strength
        foil.μs = A \ (-rhs * (foil.σs - normal_wake_ind) - buff)[:]        
        # Get the time rate of change of vorticity
        dmudt, old_mus = get_dmudt!(old_mus, foil, flow)        
        # Get the tangential component of the velocity
        qt = get_qt(foil)        
        # Add the normal component of the velocity
        qt .+=  repeat((foil.σs)',2,1) .* foil.normals  .+ normal_wake_ind'        
        # Calculate the surface pressure coefficient
        p_s = flow.ρ * sum((qt + wake_ind) .^ 2, dims=1) / 2.        
        # Calculate the unsteady pressure coefficient
        p_us = -flow.ρ.*dmudt' - flow.ρ.*(qt[1,:]'.*(-flow.Uinf .+ foil.panel_vel[1,:]' ) 
                                       .+ qt[2,:]'.*(foil.panel_vel[2,:]' ))         
        # Calculate the total pressure coefficient
        """
        ∫∞→Px1 d(∇×Ψ)/dt dC + dΦ/dt|body - (VG + VGp + (Ω×r))⋅∇Φ + 1/2||∇Φ +(∇×Ψ)|^2  = Pinf - Px1 /ρ
        """
        p = -p_s - p_us        
        coeff_p = p ./ (0.5*flow.ρ*flow.Uinf^2)        
        # Store the surface and unsteady pressure coefficients
        push!(cpus, p_us ./ (0.5*flow.ρ*flow.Uinf^2))
        push!(cps, coeff_p )
        # Calculate the edge strength
        set_edge_strength!(foil)        
        # Cancel the buffer vorticity in the wake
        cancel_buffer_Γ!(wake, foil)        
        # Move the body vorticity to the wake
        body_to_wake!(wake, foil)        
        # Update the self-induced velocity of the wake
        wake_self_vel!(wake, flow)        
        # Move the wake
        move_wake!(wake, flow)
        if i!=1 
            release_vortex!(wake, foil)
        end
         #kinematics
        (foil)(flow)
        win= (wm- .2, wm+.1)
        a = plot_current(foil, wake; window=win)
    end
    # Return the foil and wake objects
    foil, wake
end
# col = get_mdpts(foil._foil)
# 
begin   
    cps = []
    cpus = []
    foil,wake = run_sim(;steps=flow.N,  aoa = rotate(-4*pi/180)');
    # plot(foil.col[1,:], cps[end][:], yflip=true, label="steady")
    # plot!(foil.col[1,:], cpus[end][:], yflip=true, label="unsteady")
    plot(foil.col[1,:], cps[end][:], yflip=true, label="total",ls=:dash)
    # plot!(foil.col[1,:], cps[end][:] + cpus[end][:], yflip=true, label="total",ls=:dash)
    plot!(foil.col[1,:], -foil.col[2,:], label="", yflip=true)
    
    mid = foil.N ÷2
    plot(foil.col[1,1:mid], cps[end][1:mid], yflip=true, label="bottom",shape=:circle, color=:blue)        
    plot!(foil.col[1,mid+1:end], cps[end][mid+1:end], yflip=true, label="top",shape=:circle,color=:red)        
    plot!(foil.col[1,:], -foil.col[2,:], label="", yflip=true)
end

begin
    col = get_mdpts(foil._foil)
    mid = foil.N ÷2
    ΔCp = 4*sqrt.((foil.chord .- col[1,:])./col[1,:])*(0*pi/180.)
    dcp = cps[end][:]
    dcp = dcp[mid:-1:1] .- dcp[mid+1:end]
    plot(col[1,mid:-1:1], ΔCp[mid:-1:1])
    # plot!(col[1,65:end], ΔCp[65:end])
    plot!(col[1,mid+1:end], dcp)

end

begin 
    # dmudt, old_mus = get_dmudt!(old_mus, foil, flow)        
    qt = get_qt(foil) #tangential comps
    #normal comps
    tang_wake_ind = wake_ind.*foil.tangents  
    qt .+=  repeat((foil.σs)',2,1).*foil.normals  .+ normal_wake_ind'
    p_s  = flow.ρ*sum((qt + wake_ind).^2,dims=1)/2.    
    p_us = flow.ρ.*dmudt' - flow.ρ.*(qt[1,:]'.*(-flow.Uinf .+ foil.panel_vel[1,:]' .+ wake_ind[1,:]') 
                                   .+ qt[2,:]'.*(foil.panel_vel[2,:]' .+ wake_ind[2,:]'))
    # p_us = -flow.ρ.*dmudt' - flow.ρ.*(qt[1,:]'.*(-flow.Uinf .+ foil.panel_vel[1,:]' ) 
    #                               .+ qt[2,:]'.*(foil.panel_vel[2,:]' ))                                                                                   
    """
    ∫∞→Px1 d(∇×Ψ)/dt dC + dΦ/dt|body - (VG + VGp + (Ω×r))⋅∇Φ + 1/2||∇Φ +(∇×Ψ)|^2  = Pinf - Px1 /ρ
    """
    p    = -p_s - p_us
    coeff_p   = p ./ (0.5*flow.ρ*flow.Uinf^2)
    mid = foil.N ÷2
    plot(foil.col[1,1:mid], coeff_p[1:mid], yflip=true, label="bottom",shape=:circle, color=:blue)        
    plot!(foil.col[1,mid+1:end], coeff_p[mid+1:end], yflip=true, label="top",shape=:circle,color=:red)        
    plot!(foil.col[1,:], -foil.col[2,:], label="", yflip=true)

end
##BSPLINES FOR determining the dmu along the body

function is_sym(values::Vector, mid = 25)
    l,r = values[1:mid],values[end:-1:mid+1]
    @show sum(l-r), l-r
    abs(sum(l-r)) <1e-10
end
function is_sym(values::Matrix, mid = 25)
    l,r = values[:,1:mid],values[:,end:-1:mid+1]
    @show l-r
    sum(l-r) == 0
end
function is_opp(values::Matrix, mid = 25)
    l,r = values[:,1:mid],values[:,end:-1:mid+1]
    @show l+r
    sum(l+r) == 0
end
function is_opp(values::Vector, mid = 25)
    l,r = values[1:mid],values[end:-1:mid+1]
    @show l+r
    sum(l+r) == 0
end
acc_lens = cumsum(foil.panel_lengths[:])
# B = BSplineBasis(BSplineOrder(4), acc_lens[:])
S = interpolate(acc_lens, foil.μs, BSplineOrder(4))
S = interpolate(acc_lens, mu, BSplineOrder(4))
# R_n = RecombinedBSplineBasis(Derivative(1), S)
dmu = Derivative(1) * S
dmudl =  dmu.(acc_lens)
plot(acc_lens, S.(acc_lens))
plot!(acc_lens, dmu.(acc_lens),st=:scatter)
#TODO: SPLIT σ into local and induced? it is lumped into one now
qp = repeat(dmudl',2,1).*foil.tangents + repeat(foil.σs',2,1).*foil.normals
fdmu = diff(foil.μs)'./diff(acc_lens)'

dmudt, old_mus = get_dmudt!(foil, flow)
dmudt, old_mus = get_dmudt!(old_mus, foil, flow)
qp = get_qp(foil)
p_s  = -flow.ρ*sum(qp.^2,dims=1)/2.
p_us = flow.ρ.*dmudt + flow.ρ.*(qp[1,:].*(flow.Uinf ) .+ qp[2,:].*foil.panel_vel[2,:])
p    = p_s .+ p_us'
cp   = p ./ (0.5*flow.ρ*flow.Uinf^2)

dforce = repeat(-p.*foil.panel_lengths',2,1).*foil.normals
dpress = sum(dforce.*foil.panel_vel,dims=2)
force = sum(dforce,dims=2)
lift = force[1]
thrust = force[2]
power = sum(dpress, dims=1)[1]
den = 1/2 *(flow.ρ*flow.Uinf^2*foil.chord)
cforce = sum(sqrt.(force.^2))/den
clift = lift/den
cthrust = thrust/den
cpower = power/den/flow.Uinf
plot(foil.col[1,:],cp')
### SCRATHCERS
# rotating old datums - mu, phi, induced velocity
old_mus = zeros(3, foil.N)
for i = 1:2
    old_mus[i, :] .= i
end
old_mus

old_mus[2, :]
dmudt = (3 * foil.μs - 4 * old_mus[1, :] + old_mus[2, :]) / (2 *flow.Δt)
old_mus = circshift(old_mus, (1, 0))
old_mus[1,:] = foil.μs
# %% 
sten = (old_cols[2, :, :] - old_cols[1, :, :]) / (flow.Δt) #.-[flow.Uinf,0]
kine(t) = foil.kine.(foil._foil[1, :], foil.f, foil.k, t)
vy = ForwardDiff.derivative(kine, flow.Δt * 1)

yp = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, 0.02)
ym = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, 0.0)
dy = (yp - ym) / (0.02) #velocity at 0.01
# %%
#IT WORKS!


kine(x) = foil.kine(x, foil.f, foil.k, flow.n * flow.Δt)
ForwardDiff.derivative(kine, flow.Δt * 1)
ForwardDiff.derivative.(kine, foil._foil[1, :])


A, rhs = make_infs(foil, flow)
mu = A \ rhs

foil.μs = mu
release_vortex!(wake, foil)
set_edge_strength!(foil)

plot!(mu, marker=:circle)







function foil_dx_dy(N, foil)
    dx = zeros(N, N)
    dy = zeros(N, N)
    for i = 1:N
        @inbounds dx[:, i] = foil[1, :] .- foil[1, i]
        @inbounds dy[:, i] = foil[2, :] .- foil[2, i]
    end
    [dx, dy]
end
dx, dy = foil_dx_dy(N, foil)
# Find distance matrices -> first with on swimmer and then with several
#target - source
function this(N, foil)
    dx = zeros(N, N)
    dy = zeros(N, N)
    for i = 1:N
        @inbounds dx[:, i] = foil[1, :] .- foil[1, i]
        @inbounds dy[:, i] = foil[2, :] .- foil[2, i]
    end
    [dx, dy]
end
function that(N, foil)
    dx = repeat(foil[1, :], 1, N) - repeat(foil[1, :]', N, 1)
    dy = repeat(foil[2, :], 1, N) - repeat(foil[2, :]', N, 1)
    nothing
end

@btime this(N, foil) #N=2001 35.652 ms (8008 allocations: 184.69 MiB)
@btime that(N, foil) #N=2001 118.981 ms (20 allocations: 183.35 MiB)

function norms(foil)
    dxdy = diff(foil, dims=2)
    lengths = sqrt.(sum(abs2, diff(foil, dims=2), dims=1))
    tx = dxdy[1, :]' ./ lengths
    ty = dxdy[2, :]' ./ lengths
    # tangents x,y normals x, y  lengths
    return tx, ty, -ty, tx, lengths
end

tx, ty, nx, ny, lengths = norms(foil)
@assert [nx, ny] ⋅ [tx, ty] == 0.0
plot(foil[1, :], foil[2, :], aspect_ratio=:equal, marker=:circle)
plot!(foil_col[1, :], foil_col[2, :], aspect_ratio=:equal, marker=:star, color=:red)
quiver!(foil_col[1, :], foil_col[2, :], quiver=(nx[1:end], ny[1:end]))

plot(foil[1, :], foil[2, :], aspect_ratio=:equal, marker=:circle)
quiver!(foil_col[1, :], foil_col[2, :], quiver=(tx[1:end], ty[1:end]))


begin
    move = copy(foil)
    move[2, :] += h.(foil[1, :], 0.25)
    tx, ty, nx, ny, ll = norms(move)
    col = get_mdpts(move)

    plot(move[1, :], move[2, :], aspect_ratio=:equal, marker=:circle)
    plot!(col[1, :], col[2, :], aspect_ratio=:equal, marker=:star, color=:red)
    quiver!(col[1, :], col[2, :], quiver=(nx[1:end], ny[1:end]), length=0.1)
end


tx, ty, nx, ny, ll = norms(foil)
##CHANGE THE PANEL FRAMES -endpnts are the sources, col are targets 
x1 = zeros(N - 1, N - 1)
x2 = zeros(N - 1, N - 1)
y = zeros(N - 1, N - 1)
txMat = repeat(tx, N - 1, 1)
tyMat = repeat(ty, N - 1, 1)
dx = repeat(foil_col[1, :], 1, N - 1) - repeat(foil[1, 1:end-1]', N - 1, 1)
dy = repeat(foil_col[2, :], 1, N - 1) - repeat(foil[2, 1:end-1]', N - 1, 1)
# for i=1:N-1
#     x1[:,i] = (foil_col[1,:] .- foil[1,i])
#     x2[:,i] = (foil_col[2,:] .- foil[2,i])    
# end
# dx = x1
# dy = x2
x1 = dx .* txMat + dy .* tyMat
y = dx .* -tyMat + dy .* txMat

x2 = x1 - repeat(sum(diff(foil, dims=2) .* [tx; ty], dims=1), N - 1, 1)

#Grabbed from python code
#strictly for testing though y here and z there are off 
pyx1 = [[0.12596995, -0.12568028, -0.59794013, 0.84768749, 0.6282324,
        0.12983482]
    [0.50176151, 0.25039739, -0.23996686, 0.47597968, 0.25550035,
        -0.23859873]
    [0.87194163, 0.62392359, 0.12848079, 0.11474624, -0.11978317,
        -0.61264372]
    [0.86458362, 0.62057795, 0.14221534, 0.12848079, -0.1231288,
        -0.62000172]
    [0.49053864, 0.24529443, -0.2190181, 0.49692843, 0.25039739,
        -0.2498216]
    [0.12210508, -0.12743761, -0.59072591, 0.8549017, 0.62647507,
        0.12596995]]
pyx1 = reshape(pyx1, (6, 6))
pyx1' .- x1

py_z = [1.66533454e-17, -3.12043904e-02, -5.94075000e-02, -0.00000000e+00,
    5.94075000e-02, 3.12043904e-02, -1.66533454e-17]

py_z .- foil[2, :]
py_foil = [1.00000000e+00 7.50000000e-01 2.50000000e-01 0.00000000e+00 2.50000000e-01 7.50000000e-01 1.00000000e+00
    1.66533454e-17 -3.12043904e-02 -5.94075000e-02 -0.00000000e+00 5.94075000e-02 3.12043904e-02 -1.66533454e-17]

py_col = get_mdpts(py_foil)



x1, x2, y = panel_frame(py_col, py_foil)

x1, x2, y = panel_frame(foil_col, foil)
doubletMat = doublet_inf.(x1, x2, y)
sourceMat = source_inf.(x1, x2, y)
plot(x1, st=:contourf)
plot(x2, st=:contourf)
plot(y, st=:contourf)


plot!(col[1, :], col[2, :], aspect_ratio=:equal, marker=:star, color=:red)
quiver!(col[1, :], col[2, :], quiver=(nx[1:end], ny[1:end]), length=0.1)

#define the edge off of the TE
Uinf = 1.0
Δt = 0.1
#use this def to allow for implicit kutta in the future
edge_vec = Uinf * Δt * [(tx[end] - tx[1]) / 2.0, (ty[end] - ty[1]) / 2.0]
edge = [foil[:, end] foil[:, end] .+ edge_vec foil[:, end] .+ 2 * edge_vec]

plot(foil[1, :], foil[2, :], aspect_ratio=:equal, marker=:circle)
plot!(edge[1, :], edge[2, :], aspect_ratio=:equal, marker=:star, color=:green)



sigmas = sum([Uinf, 0] .* [nx; ny], dims=1)
rhs = -sourceMat * sigmas'
doubletMat \ rhs
#this all matches to now 

#make the edge doublet work
panel_frame(foil, edge)

##CHANGE THE PANEL FRAMES -endpnts are the sources, col are targets 
_, Ns = size(source)
_, Nt = size(target)
Ns -= 1 #TE accomodations
x1 = zeros(Ns, Nt)
x2 = zeros(Ns, Nt)
y = zeros(Ns, Nt)
tx, ty, nx, ny, lens = norms(source)
txMat = repeat(tx, Nt, 1)
tyMat = repeat(ty, Nt, 1)
dx = repeat(target[1, :], 1, Ns) - repeat(source[1, 1:end-1]', Nt, 1)
dy = repeat(target[2, :], 1, Ns) - repeat(source[2, 1:end-1]', Nt, 1)
x1 = dx .* txMat + dy .* tyMat
y = -dx .* tyMat + dy .* txMat
x2 = x1 - repeat(sum(diff(source, dims=2) .* [tx; ty], dims=1), Nt, 1)

edgeInf = doublet_inf.(x1, x2, y)

edgeMat = zeros(size(doubletMat))
edgeMat[:, 1] = -edgeInf[:, 1]
edgeMat[:, end] = edgeInf[:, 1]
A = doubletMat + edgeMat
μ = A \ rhs





nf = (foil._foil'*rotate(-3*pi/180)')'

plot(nf[1,:],nf[2,:])
plot!(foil._foil[1,:],foil._foil[2,:])