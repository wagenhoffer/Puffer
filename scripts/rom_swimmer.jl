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


TODOS: 
adding heaving pitching
multiple foils
nail down steady flow at TE
Theodorsen comparisons
"""

using LinearAlgebra
using Plots
using BenchmarkTools
using StaticArrays
using ForwardDiff
using BSplineKit
using Statistics
using SpecialFunctions
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

function Base.show(io::IO, w::Wake)
    print(io, "Wake x,y = ($(w.xy[1,:])\n")
    print(io, "            $(w.xy[2,:]))\n")
    print(io, "     Γ   = ($(w.Γ))")
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

function make_ang(a0=0.1; a=[0.367, 0.323, 0.310])
    a0 = a0
    a = a
    f = π
    k = 0.5

    amp(x, a) = a[1] + a[2] * x + a[3] * x^2
    h(x, f, k, t) = a0 * amp(x, a) * sin(2π * (k * x - f * t))
    h
end

function no_motion(; T=Float64)
    sig(x, f, k, t) = 0.0
end

function angle_of_attack(; aoa=5, T=Float64)
    sig(x, f, k, t) = rotation(-aoa * pi / 180)'
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
function init_params(; N=128, chord=1.0, T=Float32, motion=:make_ang, f=1, k=1, motion_parameters=nothing)
    N = N
    chord = T(chord)
    naca0012 = make_naca(N + 1; chord=chord) .|> T
    if isnothing(motion_parameters)
        kine = eval(motion)()
    else
        kine = eval(motion)(motion_parameters...)
    end
    Uinf = 1
    ρ = 1000.0
    Nt = 150
    Δt = 1 / Nt / f
    nt = 0
    δ = Uinf * Δt * 1.3
    fp = FlowParams{T}(Δt, Uinf, ρ, Nt, nt, δ)
    txty, nxny, ll = norms(naca0012)
    #TODO: 0.001 is a magic number for now
    col = (get_mdpts(naca0012) .+ repeat(0.001 .* ll, 2, 1) .* -nxny) .|> T

    edge_vec = fp.Uinf * fp.Δt * [(txty[1, end] - txty[1, 1]), (txty[2, end] - txty[2, 1])] .|> T
    edge = [naca0012[:, end] (naca0012[:, end] .+ 0.5 * edge_vec) (naca0012[:, end] .+ 1.5 * edge_vec)]
    #   kine, f, k, N, _foil,    foil ,    col,             σs, μs,         edge, chord, normals, tangents, panel_lengths,panbel_vel
    Foil{T}(kine, T(f), T(k), N, naca0012, copy(naca0012), col, zeros(T, N), zeros(T, N), edge, zeros(T, 2), 1, nxny, txty, ll[:], zeros(size(nxny))), fp
end

get_mdpts(foil) = (foil[:, 2:end] + foil[:, 1:end-1]) ./ 2

function move_edge!(foil::Foil, flow::FlowParams)
    edge_vec = [(foil.tangents[1, end] - foil.tangents[1, 1]), (foil.tangents[2, end] - foil.tangents[2, 1])]
    edge_vec ./= norm(edge_vec)
    edge_vec = flow.Uinf * flow.Δt * edge_vec
    # foil.edge = [foil.foil[:, end] (foil.foil[:, end] .+ 0.4*edge_vec) (foil.foil[:, end] .+ 1.4 * edge_vec)]
    #The edge starts at the TE -> advects some scale down -> the last midpoint
    foil.edge = [foil.foil[:, end] (foil.foil[:, end] .+ 0.5 * edge_vec) foil.edge[:, 2]]
    nothing
end

function set_collocation!(foil::Foil, S=0.009)
    foil.col = (get_mdpts(foil.foil) .+ repeat(S .* foil.panel_lengths', 2, 1) .* -foil.normals)
end

rotation(α) = [cos(α) -sin(α)
    sin(α) cos(α)]
"""
    (foil::Foil)(fp::FlowParams)

Use a functor to advance the time-step
"""
function (foil::Foil)(flow::FlowParams)
    #perform kinematics
    #TODO: only modifies the y-coordinates

    if typeof(foil.kine) == Vector{Function}
        h = foil.kine[1](foil.f, flow.n * flow.Δt)
        θ = foil.kine[2](foil.f, flow.n * flow.Δt, -π / 2)
        # foil.foil = (foil._foil'*rotation(θ))'
        rotate_about!(foil, θ)
        foil.foil[2, :] .+= h
        #Advance the foil in flow
        foil.foil .+= [-flow.Uinf, 0] .* flow.Δt .* flow.n
    else
        foil.foil[2, :] = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, flow.n * flow.Δt)
        #Advance the foil in flow
        foil.foil .+= [-flow.Uinf, 0] .* flow.Δt
    end
    norms!(foil)
    set_collocation!(foil)
    move_edge!(foil, flow)
    flow.n += 1
end


function rotate_about!(foil, θ; chord_loc=0.0)
    foil.foil = ([foil._foil[1, :] .- chord_loc foil._foil[2, :]] * rotation(θ))'
    foil.foil[1, :] .+= chord_loc
end



function make_heave_pitch(h0, θ0; T=Float64)
    θ(f, t, ψ) = θ0 * sin(2 * π * f * t + ψ)
    h(f, t) = h0 * sin(2 * π * f * t)
    [h, θ]
end
begin
    # Visualizations for heaving and pitching motions
    foil, flow = init_params(; N=50, T=Float64, motion=:make_heave_pitch,
        f=1, motion_parameters=[0.0, π / 40])
    a = plot()
    movie = @animate for i = 1:flow.N*1
        (foil)(flow)
        plot(a, foil.foil[1, :], foil.foil[2, :], aspect_ratio=:equal, label="", ylims=(-0.5, 0.5))
    end
    a
    gif(movie, "flapping.gif", fps=60)
end
begin
    # Visualizations for heaving and pitching motions
    foil, flow = init_params(; N=50, T=Float64, motion=:make_ang,
        f=1.0, k=1.0, motion_parameters=[0.1])
    a = plot()
    for i = 1:flow.N*1+1
        (foil)(flow)
        if i % 25 == 0
            plot!(a, foil.foil[1, :], foil.foil[2, :] .+ 0.25,
                aspect_ratio=:equal, label="")
        end
    end
    for i = 1:flow.N*1+1
        (foil)(flow)

        plot!(a, foil.foil[1, :] .+ 1.0, foil.foil[2, :] .- 0.25,
            aspect_ratio=:equal, label="")

    end
    a
end
"""
    get_panel_vels(foil::Foil,fp::FlowParams)

Autodiff the panel Velocities
accepts 1d waveforms and 2D prescribed kinematics
"""
function get_panel_vels(foil::Foil, fp::FlowParams)
    _t = fp.n * fp.Δt
    col = get_mdpts(foil._foil)

    if typeof(foil.kine) == Vector{Function}
        theta(t) = foil.kine[1](foil.f, t, -π / 2)
        heave(t) = foil.kine[2](foil.f, t)
        dx(t) = foil._foil[1, :] * cos(theta(t)) - foil._foil[2, :] * sin(theta(t))
        dy(t) = foil._foil[1, :] * sin(theta(t)) + foil._foil[2, :] * cos(theta(t)) .+ heave(t)
        # [ForwardDiff.derivative(t->dx(t), _t),ForwardDiff.derivative(t->dy(t), _t)]
        vels = [ForwardDiff.derivative(t -> dx(t), _t), ForwardDiff.derivative(t -> dy(t), _t)]
    else
        vy = ForwardDiff.derivative(t -> foil.
            kine.(col[1, :], foil.f, foil.k, t), _t)
        vels = [zeros(size(vy)) vy]
    end
    vels
end

function get_panel_vels!(foil::Foil, fp::FlowParams)
    _t = fp.n * fp.Δt
    col = get_mdpts(foil._foil)
    #TODO switch to a s-exp label dispatch
    if typeof(foil.kine) == Vector{Function}
        theta(t) = foil.kine[2](foil.f, t, -π / 2)
        heave(t) = foil.kine[1](foil.f, t)
        dx(t) = col[1, :] * cos(theta(t)) - col[2, :] * sin(theta(t))
        dy(t) = col[1, :] * sin(theta(t)) + col[2, :] * cos(theta(t)) .+ heave(t)

        foil.panel_vel[1, :] = ForwardDiff.derivative(t -> dx(t), _t)
        foil.panel_vel[2, :] = ForwardDiff.derivative(t -> dy(t), _t)
    else
        vy = ForwardDiff.derivative(t -> foil.
            kine.(col[1, :], foil.f, foil.k, t), _t)
        foil.panel_vel = [zeros(size(vy)) vy]'
    end
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
"""
    get_circulations(foil::Foil)

return the circulation bound to the foil and the edge panels
"""
function get_circulations(foil::Foil)
    egamma = edge_circulation(foil)
    fgamma = [-foil.μs[1], -diff(foil.μs)..., foil.μs[end]]
    fgamma, egamma
end

function make_infs(foil::Foil; ϵ=1e-10)
    x1, x2, y = panel_frame(foil.col, foil.foil)
    ymask = abs.(y) .> ϵ
    y = y .* ymask
    doubletMat = doublet_inf.(x1, x2, y)
    sourceMat = source_inf.(x1, x2, y)
    x1, x2, y = panel_frame(foil.col, foil.edge)
    edgeInf = doublet_inf.(x1, x2, y)
    edgeMat = zeros(size(doubletMat))
    edgeMat[:, 1] = -edgeInf[:, 1]
    edgeMat[:, end] = edgeInf[:, 1]
    A = doubletMat + edgeMat
    doubletMat + edgeMat, sourceMat, edgeInf[:, 2]
end

""" 
    setσ!(foil::Foil, wake::Wake,flow::FlowParams)

induced velocity from vortex wake
velocity from free stream
velocity from motion 
Tack on the wake influence outside of this function
"""
function setσ!(foil::Foil, flow::FlowParams;)
    get_panel_vels!(foil, flow)
    foil.σs = (-flow.Uinf .+ foil.panel_vel[1, :]) .* foil.normals[1, :] +
              (foil.panel_vel[2, :]) .* foil.normals[2, :]
    nothing
end



function move_wake!(wake::Wake, flow::FlowParams)
    # wake.uv = vortex_to_target(wake.xy, wake.xy, wake.xy; δ=flow.δ)
    wake.xy += wake.uv .* flow.Δt
    nothing
end

function wake_self_vel!(wake::Wake, flow::FlowParams)
    wake.uv .+= vortex_to_target(wake.xy, wake.xy, wake.Γ)
    nothing
end

"""
    body_to_wake!(wake :: Wake, foil :: Foil)

Influence of body onto the wake and the edge onto the wake
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
    wake.uv .+= [xc * foil.σs yc * foil.σs]'
    #cirulatory effects    
    # Γs = -[diff(foil.μs)... diff(foil.μ_edge) -foil.μ_edge[end]]
    # ps =  [foil.foil[:, 2:end-1] foil.edge[:,2:end]]
    # Γs = -[diff(foil.μs)... ]
    # ps =  foil.foil[:, 2:end-1]
    fg, eg = get_circulations(foil)
    Γs = [fg... eg...]
    ps = [foil.foil foil.edge]
    wake.uv .+= vortex_to_target(ps, wake.xy, Γs)
    nothing
end

function vortex_to_target(sources, targets, Γs; δ=flow.δ)
    ns = size(sources)[2]
    nt = size(targets)[2]
    vels = zeros((2, nt))
    vel = zeros(nt)
    for i = 1:ns
        dx = targets[1, :] .- sources[1, i]
        dy = targets[2, :] .- sources[2, i]
        @. vel = Γs[i] / (2π * sqrt((dx^2 + dy^2)^2 + δ^4))
        @. vels[1, :] += dy * vel
        @. vels[2, :] -= dx * vel
    end
    vels
end

function release_vortex!(wake::Wake, foil::Foil)
    # wake.xy = [wake.xy foil.edge[:, end]]
    wake.xy = [wake.xy foil.edge[:, 2]]
    wake.Γ = [wake.Γ..., (foil.μ_edge[1] - foil.μ_edge[2])]
    # Set all back to zero for the next time step
    wake.uv = [wake.uv .* 0.0 [0.0, 0.0]]
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
    wake.Γ[1] = -foil.μ_edge[end]
    nothing
end

function plot_current(foil::Foil, wake::Wake; window=nothing)
    if !isnothing(window)
        xs = (window[1], window[2])
    else
        xs = :auto
    end
    max_val = maximum(abs, wake.Γ)
    max_val = std(wake.Γ) / 2.0

    a = plot(foil.foil[1, :], foil.foil[2, :], aspect_ratio=:equal, label="")
    plot!(a, foil.edge[1, :], foil.edge[2, :], label="")
    plot!(a, wake.xy[1, :], wake.xy[2, :],
        # markersize=wake.Γ .* 10, st=:scatter, label="",
        markersize=3, st=:scatter, label="", msw=0, xlims=xs,
        marker_z=-wake.Γ, #/mean(abs.(wake.Γ)),
        color=cgrad(:coolwarm),
        clim=(-max_val, max_val))
    a
end


function plot_with_normals(foil::Foil)
    plot(foil.foil[1, :], foil.foil[2, :], aspect_ratio=:equal, label="")
    quiver!(foil.col[1, :], foil.col[2, :],
        quiver=(foil.normals[1, :], foil.normals[2, :]))
end


function get_qt(foil::Foil)
    acc_lens = 0.5 * foil.panel_lengths[1:end-1] + 0.5 * foil.panel_lengths[2:end]
    acc_lens = [0; cumsum(acc_lens)]

    # B = BSplineBasis(BSplineOrder(4), acc_lens[:])
    S = interpolate(acc_lens, foil.μs, BSplineOrder(3))
    dmu = Derivative(1) * S
    dmudl = -dmu.(acc_lens)

    #TODO: SPLIT σ into local and inducec? it is lumped into one now
    qt = repeat(dmudl', 2, 1) .* foil.tangents #+ repeat(foil.σs',2,1).*foil.normals
    qt
end

function get_phi(foil::Foil, ind_flow)
    acc_lens = 0.5 * foil.panel_lengths[1:end-1] + 0.5 * foil.panel_lengths[2:end]
    acc_lens = [0; cumsum(acc_lens)]

    # B = BSplineBasis(BSplineOrder(4), acc_lens[:])
    S = interpolate(acc_lens, ind_flow, BSplineOrder(3))
    dphi = Derivative(1) * S
    dphidl = -dmu.(acc_lens)

    #TODO: SPLIT σ into local and inducec? it is lumped into one now
    qt = repeat(dphidl, 2, 1) .* foil.tangents #+ repeat(foil.σs',2,1).*foil.normals
    qt
end


#initial allocation - not used (we don't care about the pressures at start now)
function get_dmudt!(foil::Foil, flow::FlowParams)
    old_mus = zeros(3, foil.N)
    old_mus[1, :] = foil.μs
    foil.μs / flow.Δt, old_mus
end

#after init
function get_dmudt!(old_mus, foil::Foil, flow::FlowParams)
    dmudt = (3 * foil.μs - 4 * old_mus[1, :] + old_mus[2, :]) / (2 * flow.Δt)
    old_mus = circshift(old_mus, (1, 0))
    old_mus[1, :] = foil.μs
    dmudt, old_mus
end

function get_dphidt!(oldphi, ind_flow, flow::FlowParams)
    dmudt = (3 * ind_flow - 4 * oldphi[1, :] + oldphi[2, :]) / (2 * flow.Δt)
    oldphi = circshift(oldphi, (1, 0))
    oldphi[1, :] = ind_flow
    dmudt, oldphi
end

function edge_to_body(foil::Foil)
    Γs = [-foil.μ_edge[1] foil.μ_edge[1]]
    ps = foil.edge[:, 1:2]
    vortex_to_target(ps, foil.col, Γs)
end

rotation(α) = [cos(α) -sin(α)
    sin(α) cos(α)]
begin
    # foil, flow = init_params(; N=50, T=Float64, motion=:no_motion,f=0.1, k=0.0)
    foil, flow = init_params(; N=50, T=Float64, motion=:make_heave_pitch,
        f=0.25, motion_parameters=[-0.1, π / 20])
    # foil, flow = init_params(; N=50, T=Float64, motion=:make_ang,
    # f=0.5, k=0.75,  motion_parameters=[-0.1])
    # flow.δ /=2
    aoa = rotation(8 * pi / 180)'
    foil._foil = (foil._foil' * aoa')'
    wake = Wake(foil)
    (foil)(flow)
    movie = @animate for i = 1:flow.N*5
        # begin
        A, rhs, edge_body = make_infs(foil)
        setσ!(foil, flow)
        cancel_buffer_Γ!(wake, foil)
        wake_ind = vortex_to_target(wake.xy, foil.col, wake.Γ)
        normal_wake_ind = sum(wake_ind .* foil.normals, dims=1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs*foil.σs-buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        # fully defined system, how is Kelvin?
        fg, eg = get_circulations(foil)
        # @assert sum(fg)+sum(eg)+sum(wake.Γ) <1e-15
        #DETERMINE Velocities onto the wake and move it        
        body_to_wake!(wake, foil)
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
#Plot the deviation of the last particles y position from release height
plot(foil.edge[2, 2] .- wake.xy[2, end:-1:end-150], marker=:circle)

"""
    run_sim(; steps = flow.N*10, aoa = rotate(-0*pi/180)')


TBW
"""
function run_sim(; steps=flow.N * 4, aoa=rotation(0)', motion=:no_motion,
    motion_parameters=nothing, nfoil=128, f=1)
    #TODO if passed foil, flow, wake, resume a sim
    # Initialize the foil and flow parameters
    foil, flow = init_params(; N=nfoil, T=Float64, motion=motion, f=f, motion_parameters=motion_parameters)
    # flow.δ /= 4
    den = 1 / 2 * (flow.Uinf^2 * foil.chord) # we avoid \rho unless needed
    # Rotate the foil based on the angle of attack (aoa)
    foil._foil = (foil._foil' * aoa)'
    # Create a wake object for the foil
    wake = Wake(foil)
    # Initialize the previous time step's vorticity values
    old_mus = zeros(3, foil.N)
    old_phis = zeros(3, foil.N)
    #coefficients of force, lift, thrust, power
    coeffs = zeros(4, steps)
    (foil)(flow)
    for i = 1:steps
        A, rhs, edge_body = make_infs(foil)
        setσ!(foil, flow)
        wake_ind = vortex_to_target(wake.xy, foil.col, wake.Γ)
        normal_wake_ind = sum(wake_ind .* foil.normals, dims=1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs*foil.σs-buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil)
        wake_self_vel!(wake, flow)
        p, old_mus, old_phis = panel_pressure(foil, wake_ind, old_mus, old_phis)

        coeff_p = - p ./ den
        push!(cps, coeff_p)
        ### testing
        dforce = repeat(-p .* foil.panel_lengths', 2, 1) .* foil.normals
        dpress = sum(dforce .* foil.panel_vel, dims=2)
        force = sum(dforce, dims=2)
        lift = -force[2]
        thrust = force[1]
        power = sum(dpress, dims=1)[1]

        coeffs[1, i] = sum(sqrt.(force .^ 2)) #C_forces
        coeffs[2, i] = lift    #C_lift
        coeffs[3, i] = thrust
        coeffs[4, i] = power / flow.Uinf      #C_power

        move_wake!(wake, flow)
        release_vortex!(wake, foil)
        (foil)(flow)

    end
    foil, flow,  wake, coeffs ./ den
end

function panel_pressure(foil::Foil, wake_ind, old_mus, old_phis)
    # wake_ind += edge_to_body(foil)
    normal_wake_ind = sum(wake_ind .* foil.normals, dims=1)'
    dmudt, old_mus = get_dmudt!(old_mus, foil, flow)
    dphidt, old_phis = get_dphidt!(old_phis, normal_wake_ind, flow)

    qt = get_qt(foil)
    qt .+= repeat((foil.σs)', 2, 1) .* foil.normals .+ normal_wake_ind'
    p_s = sum((qt + wake_ind) .^ 2, dims=1) / 2.0
    p_us = dmudt' + dphidt' - (qt[1, :]' .* (-flow.Uinf .+ foil.panel_vel[1, :]')
                               .+
                               qt[2, :]' .* (foil.panel_vel[2, :]'))

    # Calculate the total pressure coefficient
    """
    ∫∞→Px1 d(∇×Ψ)/dt dC + dΦ/dt|body - (VG + VGp + (Ω×r))⋅∇Φ + 1/2||∇Φ +(∇×Ψ)|^2  = Pinf - Px1 /ρ
    """
    p = p_s + p_us
    p, old_mus, old_phis
end

begin
    cps = []
    foil, flow, wake, coeffs = run_sim(; steps=flow.N * 3, aoa=rotation(-5 * pi / 180)', motion=:no_motion, nfoil=64, f=0.015)
    # foil, wake,coeffs = run_sim(;steps=flow.N*3, f=0.25, motion=:make_heave_pitch, motion_parameters=[0.0, π/20], nfoil=64);
    # plot(foil.col[1,:], cps[end][:], yflip=true, label="steady")

    plot(foil.col[1, :], cps[end][:], yflip=true, label="total", ls=:dash)
    # plot!(foil.col[1,:], cps[end][:] + cpus[end][:], yflip=true, label="total",ls=:dash)
    plot!(foil.col[1, :], -foil.col[2, :], label="", yflip=true)

    mid = foil.N ÷ 2
    plot(foil.col[1, 1:mid+1], cps[end][1:mid+1], yflip=true, label="bottom", shape=:circle, color=:blue)
    plot!(foil.col[1, mid+1:end], cps[end][mid+1:end], yflip=true, label="top", shape=:circle, color=:red)
    plot!(foil.col[1, :], -foil.col[2, :], label="", yflip=true)
end

begin
    col = get_mdpts(foil._foil)
    mid = foil.N ÷ 2
    ΔCp = 4 * sqrt.((foil.chord .- col[1, :]) ./ col[1, :]) * (0 * pi / 180.0)
    dcp = cps[end][:]
    dcp = dcp[mid:-1:1] .- dcp[mid+1:end]
    plot(col[1, mid:-1:1], ΔCp[mid:-1:1])
    # plot!(col[1,65:end], ΔCp[65:end])
    plot!(col[1, mid+1:end], dcp)

end

begin
    ##Theodorsen comparisons
    # Load input parameters
    N_STEP = 150
    N_CYC = 3

    tau = range(0, length=N_STEP * N_CYC, stop=N_CYC)

    # Define constants
    b = 0.5 * foil.chord
    a = -0.25
    w = 2.0 * pi * foil.f
    t = tau
    RHO = flow.ρ
    U = abs(1)
    theta_max = θ0 = π / 20
    heave_max = 0.1 * foil.chord
    PHI = π / 2

    k = w * b / U
    k2 = π * foil.f * foil.chord / U
    St = 2.0 * foil.f * heave_max / U

    F = (besselj1(k) * (besselj1(k) + bessely0(k)) + bessely1(k) * (bessely1(k) - besselj0(k))) / ((besselj1(k) + bessely0(k))^2 + (bessely1(k) - besselj0(k))^2)
    G = -(bessely1(k) * bessely0(k) + besselj1(k) * besselj0(k)) / ((besselj1(k) + bessely0(k))^2 + (bessely1(k) - besselj0(k))^2)
    L = zeros(size(t))
    L2 = zeros(size(t))
    @. L = -RHO * b^2 * (U * pi * theta_max * w * cos(w * t + PHI) - pi * heave_max * w^2 * sin(w * t) + pi * b * a * theta_max * w^2 * sin(w * t + PHI)) -
           2.0 * pi * RHO * U * b * F * (U * theta_max * sin(w * t + PHI) + heave_max * w * cos(w * t) + b * (0.5 - a) * theta_max * w * cos(w * t + PHI)) -
           2.0 * pi * RHO * U * b * G * (U * theta_max * cos(w * t + PHI) - heave_max * w * sin(w * t) - b * (0.5 - a) * theta_max * w * sin(w * t + PHI))
    @. L2 = RHO * U^2 * foil.chord * sqrt(R^2 + I^2) * real(exp(1im * w * tau))
    Cl = real.(L) / (0.5 * RHO * U^2 * foil.chord)
    L2 ./= (0.5 * RHO * U^2 * foil.chord)

    # Plot the results
    plot(tau, Cl, label="Theodorsen")
    plot!(tau, L2)

end
begin
    θ0 = π / 20
    foil, flow, wake, coeffs = run_sim(; steps=flow.N * 3, f=1.0, 
    motion=:make_heave_pitch, motion_parameters=[0.0, θ0], nfoil=64)
    ##GARRICK model
    t = flow.Δt:flow.Δt:flow.N*N_CYC*flow.Δt
    ω = 2π * foil.f
    a = -1 #-1 is rotation about leading edge
    theo = 1im * hankelh1(1, k) / (hankelh1(0, k) + 1im * hankelh1(1, k))
    F, G = real(theo), imag(theo)
    R =  π * θ0 * (k^2 / 2 * (0.125 + a^2) + (0.5 + a) * (F - (0.5 - a) * k * G))
    I = -π * θ0 * (k / 2 * (0.5 - a) - (0.5 + a) * (G + (0.5 - a) * k * F))
    Φ = atan(I, R)
    gar = sqrt.(R .^ 2 + I .^ 2) # *  0.5* flow.ρ* flow.Uinf^2* foil.chord* 
    Cl = gar .* exp.(1im * ω * t)
    M = gar .* exp.(1im * ω * t .+ Φ)
    plot(t, real(Cl), label="Garrick  \$C_L\$")
    plot!(t[50:end], coeffs[2, 50:end]/2., label ="BEM  \$C_L\$" )
end
##BSPLINES FOR determining the dmu along the body
