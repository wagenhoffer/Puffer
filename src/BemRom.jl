# module BemRom
# __precompile__(false)
""" EXPORT EVERYTHING FOR THE LAZY"""

export FlowParams, Foil, Wake, make_naca, make_waveform, make_ang, make_heave_pitch, no_motion, norms, norms!,
        source_inf, doublet_inf, init_params, get_mdpts, move_edge!, set_collocation!, 
        rotate_about!, get_panel_vels, get_panel_vels!, panel_frame, 
        edge_circulation, get_circulations, make_infs, setσ!, move_wake!, body_to_wake!, 
        vortex_to_target, release_vortex!, set_edge_strength!, 
        cancel_buffer_Γ!, plot_current, plot_with_normals,
        get_qt,get_phi, get_dmudt!, get_dphidt!, edge_to_body, run_sim, rotation, 
        get_performance, panel_pressure, wake_self_vel!,time_increment!

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
    Ncycles::Int #number of cycles
    n::Int #current time step
    δ::T #desing radius
end

mutable struct Foil{T} <: Body
    kine  #kinematics: heave or traveling wave function
    f::T #wave freq
    k::T # wave number
    N::Int # number of elements
    _foil::Matrix{T} # coordinates in the Body Frame
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
    wake_ind_vel::Matrix{T} #velocity induced by wake at collocation points
    ledge::Matrix{T} #leading edge panel 
    μ_ledge::Vector{T} #leading edge doublet strength
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
                         + 2 * z * (atan(z, x2) - atan(z, x1))) / (4π)
doublet_inf(x1, x2, z) = -(atan(z, x2) - atan(z, x1)) / (2π)

"""
    init_params(;N=128, chord = 1.0, T=Float32 )

TODO Add wake stuff
"""
function init_params(; kwargs...)
	T = kwargs[:T]
    N = kwargs[:N]
	chord = T(kwargs[:chord])
	naca0012 = make_naca(N + 1; chord = chord) .|> T
	if haskey(kwargs,:motion_parameters)
		kine = eval(kwargs[:kine])(kwargs[:motion_parameters]...)
	else
		kine = eval(kwargs[:kine])()
	end
	Uinf = kwargs[:Uinf]
	ρ = kwargs[:ρ]
	Nt = kwargs[:Nt]
    f = kwargs[:f]
	Δt = 1 / Nt / f
	nt = 0
	δ = Uinf * Δt * 1.3
	fp = FlowParams{T}(Δt, Uinf, ρ, Nt, kwargs[:Ncycles], nt, δ)
	
    txty, nxny, ll = norms(naca0012)
	
	col = (get_mdpts(naca0012) .+ repeat(0.001 .* ll, 2, 1) .* -nxny) .|> T

	edge_vec = Uinf * Δt * [(txty[1, end] - txty[1, 1]), (txty[2, end] - txty[2, 1])] .|> T
	edge = [naca0012[:, end] (naca0012[:, end] .+ 0.5 * edge_vec) (naca0012[:, end] .+ 1.5 * edge_vec)]
	            #  kine, f,    k,             N,    _foil,    foil ,       col,          σs,          μs, edge,   μ_edge,chord,normals, tangents, panel_lengths, panbel_vel, wake_ind_vel, ledge, μ_ledge
	foil = Foil{T}(kine, T(f), T(kwargs[:k]), N, naca0012, copy(naca0012), col, zeros(T, N), zeros(T, N), edge, zeros(T, 2), 1, nxny, txty, ll[:], zeros(size(nxny)),zeros(size(nxny)),zeros(T, 3,2),zeros(T,2))
    foil, fp
end


defaultDict = Dict(:T     => Float64,
	:N     => 64,
	:kine  => :make_ang,
	:f     => 1,
	:k     => 1,
	:chord => 1.0,
	:Nt    => 150,
    :Ncycles  => 1,
	:Uinf  => 1.0,
	:ρ     => 1000.0, 
    :Nfoils => 1)

get_mdpts(foil) = (foil[:, 2:end] + foil[:, 1:end-1]) ./ 2

function move_edge!(foil::Foil, flow::FlowParams)
    edge_vec = [(foil.tangents[1, end] - foil.tangents[1, 1]), (foil.tangents[2, end] - foil.tangents[2, 1])]
    edge_vec ./= norm(edge_vec)
    edge_vec .*= flow.Uinf * flow.Δt 
    #The edge starts at the TE -> advects some scale down -> the last midpoint
    foil.edge = [foil.foil[:, end] (foil.foil[:, end] .+ 0.4 * edge_vec) foil.edge[:, 2]]
    #static buffer is a bugger
    # foil.edge = [foil.foil[:, end] (foil.foil[:, end] .+ 0.4 * edge_vec) (foil.foil[:, end] .+ 1.4 * edge_vec)]
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
    if typeof(foil.kine) == Vector{Function}
        h = foil.kine[1](foil.f, flow.n * flow.Δt)
        θ = foil.kine[2](foil.f, flow.n * flow.Δt, -π/2)
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

"""
    get_panel_vels(foil::Foil,fp::FlowParams)

Autodiff the panel Velocities
accepts 1d waveforms and 2D prescribed kinematics
"""
function get_panel_vels(foil::Foil, fp::FlowParams)
    _t = fp.n * fp.Δt
    col = get_mdpts(foil._foil)

    if typeof(foil.kine) == Vector{Function}
        theta(t) = foil.kine[1](foil.f, t, - π / 2)
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
        theta(t) = foil.kine[2](foil.f, t, -π/2)
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

""" flow.n += 1
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
    doubletMat + edgeMat, sourceMat, edgeInf[:, 2]
end

function make_infs(foils::Vector{Foil{T}}; ϵ=1e-10) where T <: Real
    nfoils = length(foils)
    Ns = [foil.N for foil in foils]
    N = sum(Ns)
    Ns = [0 cumsum(Ns)...]
    doubletMat = zeros(N, N)
    sourceMat  = zeros(N, N)
    edgeMat    = zeros(N, N)
    #assumes all foils are same size
    buffers    = zeros(nfoils, foils[1].N)
    #for pushing
    # buffers = []
    for i = 1:nfoils
        for j = 1:nfoils
            x1, x2, y = panel_frame(foils[i].col, foils[j].foil)
            # ymask = abs.(y) .> ϵ
            # y = y .* ymask
            doubletMat[Ns[i]+1:Ns[i+1],Ns[j]+1:Ns[j+1]] = doublet_inf.(x1, x2, y)
            sourceMat[Ns[i]+1:Ns[i+1],Ns[j]+1:Ns[j+1]] = source_inf.(x1, x2, y)
            if i == j 
                nn = foils[i].N
                x1, x2, y = panel_frame(foils[i].col, foils[i].edge)
                edgeInf = doublet_inf.(x1, x2, y)
                buffers[i,:] = edgeInf[:,2]
                # push!(buffers, edgeInf[:, 2])
                edgeMat[Ns[i]+1:Ns[i+1], Ns[i]+1] = -edgeInf[:, 1]
                edgeMat[Ns[i]+1:Ns[i+1], Ns[i+1]] = edgeInf[:, 1]
            end
        end 
    end                            
    A = doubletMat + edgeMat
    doubletMat + edgeMat, sourceMat, buffers'
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
    wake.xy += wake.uv .* flow.Δt
    nothing
end

function wake_self_vel!(wake::Wake, flow::FlowParams)
    wake.uv .+= vortex_to_target(wake.xy, wake.xy, wake.Γ, flow)
    nothing
end
::Foil
"""
    body_to_wake!(wake :: Wake, foil :: Foil)

Influence of body onto the wake and the edge onto the wake
"""
function body_to_wake!(wake::Wake, foil::Foil,flow::FlowParams)
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
    fg, eg = get_circulations(foil)
    Γs = [fg... eg...]
    ps = [foil.foil foil.edge]
    wake.uv .+= vortex_to_target(ps, wake.xy, Γs, flow)
    nothing
end

function vortex_to_target(sources, targets, Γs, flow)
    ns = size(sources)[2]
    nt = size(targets)[2]
    vels = zeros((2, nt))
    vel = zeros(nt)
    for i = 1:ns
        dx = targets[1, :] .- sources[1, i]
        dy = targets[2, :] .- sources[2, i]
        @. vel = Γs[i] / (2π * sqrt((dx^2 + dy^2)^2 + flow.δ^4))
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

    if any(foil.μ_ledge .!= 0)
        wake.xy = [wake.xy foil.ledge[:,2]]
        wake.Γ = [wake.Γ..., (foil.μ_ledge[1] - foil.μ_ledge[2])]
        wake.uv = [wake.uv .* 0.0 [0.0, 0.0]]
    end
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
function set_ledge_strength!(foil::Foil)
    """Assumes that foil.μs has been set for the current time step 
        TODO: Extend to perform streamline based Kutta condition
    """
    mid = foil.N ÷ 2
    foil.μ_ledge[2] = foil.μ_ledge[1]
    foil.μ_ledge[1] = foil.μs[mid] - foil.μs[mid+1]
    nothing
end
function cancel_buffer_Γ!(wake::Wake, foil::Foil)
    #TODO : Add iterator for matching 1->i for nth foil
    wake.xy[:, 1] = foil.edge[:, end]
    wake.Γ[1] = -foil.μ_edge[end]
    #LESP
    if foil.μ_ledge[2] != 0
        wake.xy[:, 2] = foil.ledge[:, end]
        wake.Γ[2] = -foil.μ_ledge[end]
    end
    nothing
end

function plot_current(foil::Foil, wake::Wake; window=nothing)
    if !isnothing(window)
        xs = (window[1], window[2])
    else
        xs = :auto
    end
    max_val = maximum(abs, wake.Γ)A = A + le_inf
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

function get_phi(foil::Foil, wake::Wake)
    phi = zeros(foil.N)           
    for i = 1:size(wake.Γ)[1]
        dx = foil.col[1, :] .- wake.xy[1, i]
        dy = foil.col[2, :] .- wake.xy[2, i]
        @. phi = -wake.Γ[i] *mod2pi(atan(dy,dx))/(2π)        
    end
    phi
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

function get_dphidt!(oldphi, phi, flow::FlowParams)
    dmudt = (3 * phi - 4 * oldphi[1, :] + oldphi[2, :]) / (2 * flow.Δt)
    oldphi = circshift(oldphi, (1, 0))
    oldphi[1, :] = phi
    dmudt, oldphi
end

function edge_to_body(foil::Foil, flow::FlowParams)
    Γs = [-foil.μ_edge[1] foil.μ_edge[1]]
    ps = foil.edge[:, 1:2]
    vortex_to_target(ps, foil.col, Γs, flow)
end

"""
    run_sim(; steps = flow.N*10, aoa = rotate(-0*pi/180)')


TBW
"""
function run_sim(; kwargs...)
    # Initialize the foil and flow parameters
    foil, flow = init_params(; kwargs...)
    steps = flow.N * flow.Ncycles
    # flow.δ /= 4
    den = 1 / 2 * (flow.Uinf^2 * foil.chord) # we avoid \rho unless needed
    # Rotate the foil based on the angle of attack (aoa)
    if haskey(kwargs, :aoa)
        foil._foil = (foil._foil' * rotation(kwargs[:aoa])')'            
    end
    
    # Create a wake object for the foil
    wake = Wake(foil)
    # Initialize the previous time step's vorticity values
    old_mus = zeros(3, foil.N)::Foil
    old_phis = zeros(3, foil.N)
    #coefficients of force, lift, thrust, power
    coeffs = zeros(4, steps)
    (foil)(flow)
    for i = 1:steps
        A, rhs, edge_body = make_infs(foil)
        setσ!(foil, flow)
        foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
        normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'
        foil.σs -= normal_wake_ind[:]
        buff = edge_body * foil.μ_edge[1]
        foil.μs = A \ (-rhs*foil.σs - buff)[:]
        set_edge_strength!(foil)
        cancel_buffer_Γ!(wake, foil)
        body_to_wake!(wake, foil, flow)
        wake_self_vel!(wake, flow)
        phi =  get_phi(foil, wake)
        p, old_mus, old_phis = panel_pressure(foil, flow, old_mus, old_phis, phi)
        coeffs[:,i] .= get_performance(foil, flow, p)
             
        move_wake!(wake, flow)
        release_vortex!(wake, foil)
        (foil)(flow)

    end
    foil, flow,  wake, coeffs 
end


function get_performance(foil, flow, p)
    dforce = repeat(-p .* foil.panel_lengths', 2, 1) .* foil.normals
    dpress = sum(dforce .* foil.panel_vel, dims=2)
    force  = sum(dforce, dims=2)
    lift   = -force[2]
    thrust = force[1]
    power  = sum(dpress, dims=1)[1]

    [sum(sqrt.(force .^ 2)), #C_forces
    lift,    #C_lift
    thrust,
    power / flow.Uinf]      #C_power    
end

function panel_pressure(foil::Foil, flow,  old_mus, old_phis, phi)
    # wake_ind += edge_to_body(foil, flow)
    normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'

    dmudt, old_mus = get_dmudt!(old_mus, foil, flow)
    dphidt, old_phis = get_dphidt!(old_phis, phi, flow)

    qt = get_qt(foil)
    qt .+= repeat((foil.σs)', 2, 1) .* foil.normals 
    p_s = sum((qt + foil.wake_ind_vel) .^ 2, dims=1) / 2.0
    p_us = dmudt' + dphidt' - (qt[1, :]' .* (-flow.Uinf .+ foil.panel_vel[1, :]')
                               .+
                               qt[2, :]' .* (foil.panel_vel[2, :]'))

    # Calculate the total pressure coefficient
    """
    ∫∞→Px1 d(∇×Ψ)/dt dC + dΦ/dt|body - (VG + VGp + (Ωxr))⋅∇Φ + 1/2||∇Φ +(∇×Ψ)|^2  = Pinf - Px1 /ρ
    """
    p = p_s + p_us
    p, old_mus, old_phis
end

"""
    time_increment!(flow::FlowParams, foil::Foil, wake::Wake)

does a single timestep of the simulation. This is the core of the simulation.
Useful for debugging or for plotting purposes.

Reorder the operations so it ends with a fully defined system. This way we can
grab metrics of the foil. 
"""
function time_increment!(flow::FlowParams, foil::Foil, wake::Wake)
    if flow.n != 1
        move_wake!(wake, flow)   
        release_vortex!(wake, foil)
    end    
    (foil)(flow)
    
    A, rhs, edge_body = make_infs(foil)
    setσ!(foil, flow)    
    foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
    normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'
    foil.σs -= normal_wake_ind[:]
    buff = edge_body * foil.μ_edge[1]
    foil.μs = A \ (-rhs*foil.σs-buff)[:]
    set_edge_strength!(foil)
    cancel_buffer_Γ!(wake, foil)
    body_to_wake!(wake, foil, flow)
    wake_self_vel!(wake, flow)    
    nothing
end

# end


# #### TODO LIST WHAT NEEDS TO BE VEC'D FOR MULTI and what can LOOP
# A, rhs, edge_body = make_infs(foils) #̌#CHECK
# [setσ!(foil, flow) for foil in foils] #CHECK
# σs = [] #CHECK
# buff = []
# for (i, foil) in enumerate(foils)
#     foil.wake_ind_vel = vortex_to_target(wake.xy, foil.col, wake.Γ, flow)
#     normal_wake_ind = sum(foil.wake_ind_vel .* foil.normals, dims=1)'
#     foil.σs -= normal_wake_ind[:]
#     push!(σs, foil.σs...)
#     push!(buff, (edge_body[:,i] * foil.μ_edge[1])...)    
# end
# μs =  A \ (-rhs*σs - buff)
# for (i, foil) in enumerate(foils)
#     foil.μs = μs[(i-1)*foil.N+1:i*foil.N]
# end
# set_edge_strength!.(foils)
# cancel_buffer_Γ!(wake, foil)
# body_to_wake!(wake, foil, flow)
# wake_self_vel!(wake, flow)
# phi =  get_phi(foil, wake)
# p, old_mus, old_phis = panel_pressure(foil, flow, wake_ind, old_mus, old_phis, phi)
# coeffs[:,i] .= get_performance(foil, flow, p)
     
# move_wake!(wake, flow)
# release_vortex!(wake, foil)
# (foil)(flow)
