module BEM_ROM
__precompile__(false)

using LinearAlgebra
using Plots
using BenchmarkTools
using StaticArrays
using ForwardDiff
""" EXPORT EVERYTHING FOR THE LAZY"""
export FlowParams, Foil, Wake, make_naca, make_waveform, make_ang, no_motion, norms, norms!,
        source_inf, doublet_inf, init_params, get_mdpts, move_edge!, set_collocation!, get_panel_vels, panel_frame, 
        edge_circulation, setσ!, move_wake!, body_to_wake!, vortex_to_target, release_vortex!, set_edge_strength!, 
        cancel_buffer_Γ!, plot_current, plot_with_normals, run_sim, rotate
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
    f = k = 1

    amp(x, a) = a[1] + a[2] * x + a[3] * x^2
    h(x, f, k, t) = a0 * amp(x, a) * sin(2π * (k * x - f * t))
    # h(x,t) = f,k -> h(x,f,k,t)
    h
end
function no_motion(;T=Float64)
    sig(x,f,k,t) = 0.0
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
function init_params(; N=128, chord=1.0, T=Float32, motion=:make_ang)
    N = N
    chord = T(chord)
    naca0012 = make_naca(N + 1; chord=chord) .|> T
    
    ang = eval(motion)()
    fp = FlowParams{T}(1.0 / 150.0, 1, 1000.0, 150, 0, 0.013)
    txty, nxny, ll = norms(naca0012)
    #TODO: 0.001 is a magic number for now
    col = (get_mdpts(naca0012) .+  repeat(0.001 .*ll,2,1).*-nxny).|> T
    
    edge_vec = fp.Uinf * fp.Δt * [(txty[1, end] - txty[1, 1]) / 2.0, (txty[2, end] - txty[2, 1]) / 2.0] .|> T
    edge = [naca0012[:, end] (naca0012[:, end] .+ edge_vec) (naca0012[:, end] .+ 2 * edge_vec)]
    #   kine, f, k, N, _foil,    foil ,    col,             σs, μs,         edge, chord, normals, tangents, panel_lengths
    Foil{T}(ang, T(1), T(1), N, naca0012, copy(naca0012), col, zeros(T, N), zeros(T, N), edge, zeros(T, 2), 1, nxny, txty, ll[:]), fp
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
    foil.foil[2, :] = foil._foil[2, :] .+ foil.kine.(foil._foil[1, :], foil.f, foil.k, flow.n * flow.Δt)
    #Advance the foil in flow
    foil.foil .+= [-flow.Uinf, 0] .* flow.Δt
    # foil.col = get_mdpts(foil.foil)
    norms!(foil)
    set_collocation!(foil)    
    move_edge!(foil, flow)
    flow.n += 1
    # print(foil.foil[2,:] == foil._foil[2,:])
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

function make_infs(foil::Foil, flow::FlowParams; ϵ=1e-10)
    x1, x2, y = panel_frame(foil.col, foil.foil)
    ymask = abs.(y) .> ϵ
    # y = y .* ymask
    doubletMat = doublet_inf.(x1, x2, y)
    sourceMat = source_inf.(x1, x2, y)
    x1, x2, y = panel_frame(foil.col, foil.edge)
    edgeInf = doublet_inf.(x1, x2, y)
    edgeMat = zeros(size(doubletMat))
    edgeMat[:, 1] = -edgeInf[:, 1]
    edgeMat[:, end] = edgeInf[:, 1]
    A = doubletMat + edgeMat
    # A, sourceMat
    doubletMat + edgeMat, sourceMat
end

""" 
    setσ!(foil::Foil, wake::Wake,flow::FlowParams)

induced velocity from vortex wake
velocity from free stream
velocity from motion 
"""
function setσ!(foil::Foil, wake::Wake, flow::FlowParams)

    wake_ind = vortex_to_target(wake.xy,foil.col,wake.Γ)

    #no body motion in the x-dir ...yet #TODO

    vy = ForwardDiff.derivative(t -> foil.kine.(foil._foil[1, :], foil.f, foil.k, t), flow.Δt * flow.n)
    vy = (vy[1:end-1] + vy[2:end]) / 2.0 #averaging, ugh...
    foil.σs = (-flow.Uinf .+ wake_ind[1,:]) .* foil.normals[1, :] +
              (vy + wake_ind[1,:])  .* foil.normals[2, :]
    zeros(size(vy)),vy
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
    #TODO: cancellation of the buffer issues
    Γs = -[diff(foil.μs)... diff(foil.μ_edge)... -foil.μ_edge[end]]
    ps = hcat(foil.foil[:, 2:end-1], foil.edge[:, 2:end])
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
    wake.xy = [wake.xy foil.edge[:, end]]
    wake.Γ = [wake.Γ..., foil.μ_edge[1] - foil.μ_edge[2]]
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
    wake.Γ[1] = -foil.μ_edge[end]
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
plot_current(foil,wake;window= (minimum(foil.foil[1,:]')-foil.chord/2.0, maximum(foil.foil[1,:])+foil.chord*5))
plot_current(foil,wake;window=(-2.2,-1.8))
                                

function plot_with_normals(foil::Foil)
    plot(foil.foil[1, :], foil.foil[2, :], aspect_ratio=:equal, label="")
    quiver!(foil.col[1, :], foil.col[2, :],
        quiver=(foil.normals[1, :], foil.normals[2, :]))
end


rotate(α) = [cos(α) -sin(α)
             sin(α)  cos(α)]


end # module
