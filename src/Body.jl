abstract type Body end

#  Foil struct and related functions
mutable struct Foil{T} <: Body
    kine::Any  #kinematics: heave or traveling wave function
    f::T #wave freq
    k::T # wave number
    ψ::T # phase
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
    pivot::T # where the foil pivots about as a fraction of chord
    LE::Vector{T} #explicit position of leading edge
    θ::T # orientation of foil
    vel::Vector{T} #velocity of body <- self propelled systems
end

# NACA0012 Foil
function make_naca(N; chord = 1, thick = 0.12)
    # N = 7
    an = [0.2969, -0.126, -0.3516, 0.2843, -0.1036]
    # T = 0.12 #thickness
    function yt(x_)
        thick / 0.2 *
        (an[1] * x_^0.5 + an[2] * x_ + an[3] * x_^2 + an[4] * x_^3 + an[5] * x_^4)
    end
    #neutral x
    x = (1 .- cos.(LinRange(0, pi, (N + 2) ÷ 2))) / 2.0
    foil = [[x[end:-1:1]; x[2:end]]';
        [-yt.(x[end:-1:1]); yt.(x[2:end])]']
    foil .* chord
end

function make_teardrop(N; chord = 1, thick = 0.01)
    # Generate points for the bottom surface
    xb = LinRange(pi, 0, (N + 2) ÷ 2)
    xt = LinRange(0, pi, (N + 2) ÷ 2)

    # Slopes and intersects for the line segments
    m = -thick / 2 / (chord - thick / 2)
    b = thick / 2 + thick^2 / 4 / (chord - thick / 2)

    # Tear drop shape equation.
    x_c = 0.5 .* (1 .- cos.(xb))
    xb = x_c .* chord
    xb1 = filter(x -> x <= thick / 2, xb)
    xb2 = filter(x -> x > thick / 2, xb)

    zb2 = -m .* xb2 .- b
    zb1 = -sqrt.((thick / 2)^2 .- (xb1 .- thick / 2) .^ 2)
    zb = vcat(zb2, zb1)

    # Tear drop shape equation.
    x_c = 0.5 .* (1 .- cos.(xt))
    xt = x_c .* chord
    xt1 = filter(x -> x <= thick / 2, xt)
    xt2 = filter(x -> x > thick / 2, xt)

    zt1 = sqrt.((thick / 2)^2 .- (xt1 .- thick / 2) .^ 2)
    zt2 = m .* xt2 .+ b
    zt = vcat(zt1, zt2)

    zb[1] = 0
    zt[1] = 0
    zb[end] = 0

    # Merge top and bottom surfaces together
    x = vcat(xb, xt[2:end])
    z = vcat(zb, zt[2:end])

    [x'; z']
end

function make_vandevooren(N; chord = 1.0, thick = 0.75, K = 1.93)
    A = chord * ((1 + thick)^(K - 1)) * (2^(-K))
    THETA = LinRange(0, pi, Int(N ÷ 2) + 1)

    R1 = sqrt.((A * cos.(THETA) .- A) .^ 2 + (A^2) .* sin.(THETA) .^ 2)
    R2 = sqrt.((A * cos.(THETA) .- thick * A) .^ 2 + (A^2) .* sin.(THETA) .^ 2)

    THETA1 = atan.(A * sin.(THETA), A * cos.(THETA) .- A)
    THETA2 = atan.(A * sin.(THETA), A * cos.(THETA) .- thick * A)

    x = ((R1 .^ K) ./ (R2 .^ (K - 1))) .* (cos.(K * THETA1) .* cos.((K - 1) * THETA2) +
         sin.(K * THETA1) .* sin.((K - 1) * THETA2))
    z_top = ((R1 .^ K) ./ (R2 .^ (K - 1))) .* (sin.(K * THETA1) .* cos.((K - 1) * THETA2) -
             cos.(K * THETA1) .* sin.((K - 1) * THETA2))
    z_bot = -((R1 .^ K) ./ (R2 .^ (K - 1))) .* (sin.(K * THETA1) .* cos.((K - 1) * THETA2) -
             cos.(K * THETA1) .* sin.((K - 1) * THETA2))

    x = x .- x[end]  # Carrying the leading edge to the origin
    x[1] = chord

    z_top[1] = 0
    z_bot[1] = 0
    z_bot[end] = 0

    # Merge top and bottom surfaces together
    x = vcat(x, x[(end - 1):-1:1])
    z = vcat(z_bot, z_top[(end - 1):-1:1])

    [x'; z']
end

function make_wave(a0 = 0.1; a = [0.367, 0.323, 0.310])
    a0 = a0
    a = a
    f = π
    k = 0.5

    amp(x, a) = a[1] + a[2] * x + a[3] * x^2
    h(x, f, k, t, ψ) = a0 * amp(x, a) * sin(2π * (k * x - f * t) + ψ)
    h
end

make_ang(a0 = 0.1; a = [0.367, 0.323, 0.310]) = make_wave(a0; a = a)
make_car(a0) = make_wave(a0; a=[0.2,-0.825, 1.625])

function make_heave_pitch(h0, θ0; T = Float64)
    θ(f, t, ψ) = θ0 * sin(2 * π * f * t + ψ)
    h(f, t) = h0 * sin(2 * π * f * t)
    [h, θ]
end

function make_eldredge(α, αdot; s = 0.001, chord = 1.0, Uinf = 1.0)
    K = αdot * chord / (2 * Uinf)
    a(σ) = π^2 * K / (2 * α * (1.0 - σ))

    t1 = 1.0
    t2 = t1 + α / 2 / K
    t3 = t2 + π * α / 4 / K - α / 2 / K
    t4 = t3 + α / 2 / K
    function eld(t)
        log((cosh(a(s) * (t - t1)) * cosh(a(s) * (t - t4))) /
            (cosh(a(s) * (t - t2)) * cosh(a(s) * (t - t3))))
    end
    maxG = maximum(filter(x -> !isnan(x), eld.(0:0.1:100)))
    pitch(f, tt, p) = α * eld(tt) / maxG
    heave(f, t) = 0.0
    [heave, pitch]
end

function no_motion(; T = Float64)
    sig(x, f, k, t,ψ) = 0.0
end

function angle_of_attack(; aoa = 5, T = Float64)
    sig(x, f, k, t) = rotation(-aoa * pi / 180)'
end

function norms(foil)
    dxdy = diff(foil, dims = 2)
    lengths = sqrt.(sum(abs2, diff(foil, dims = 2), dims = 1))
    tx = dxdy[1, :]' ./ lengths
    ty = dxdy[2, :]' ./ lengths
    # tangents x,y normals x, y  lengths
    return [tx; ty], [-ty; tx], lengths
end

function norms!(foil::Foil)
    dxdy = diff(foil.foil, dims = 2)
    lengths = sqrt.(sum(abs2, dxdy, dims = 1))
    tx = dxdy[1, :]' ./ lengths
    ty = dxdy[2, :]' ./ lengths
    # tangents x,y normals x, y  lengths
    foil.tangents = [tx; ty]
    foil.normals = [-ty; tx]
    foil.panel_lengths = [lengths...]
    nothing
end

function move_edge!(foil::Foil, flow::FlowParams; startup = false, edge_length= 0.4)
    edge_vec = [
        (foil.tangents[1, end] - foil.tangents[1, 1]),
        (foil.tangents[2, end] - foil.tangents[2, 1]),
    ]
    edge_vec ./= norm(edge_vec)
    edge_vec .*= flow.Uinf * flow.Δt
    #The edge starts at the TE -> advects some scale down -> the last midpoint
    foil.edge = [foil.foil[:, end] (foil.foil[:, end] .+  edge_length * edge_vec) foil.edge[:, 2]]
    foil.edge = [foil.foil[:, end] (foil.foil[:, end] .+  edge_length * edge_vec) (foil.foil[:, end] .+ (1.0 + edge_length)* edge_vec)]
    if startup
        foil.edge = [foil.foil[:, end] (foil.foil[:, end] .+  edge_length * edge_vec) (foil.foil[:, end] .+ (1.0 + edge_length)* edge_vec)]
    end
    nothing
end

function set_collocation!(foil::Foil, S = 0.005)
    foil.col = (get_mdpts(foil.foil) .+
                repeat(S .* foil.panel_lengths', 2, 1) .* -foil.normals)
end

rotation(α) = [cos(α) -sin(α)
              sin(α) cos(α)]

function next_foil_pos(foil::Foil, flow::FlowParams)
    #TODO: rework for self propelled swimming
    #perform kinematics
    Δxy = flow.Uinf .* flow.Δt
    LE = foil.LE
    xle1 = [cos(foil.θ + π), sin(foil.θ + π)] .* Δxy
    LE += xle1[:]
    if typeof(foil.kine) == Vector{Function}
        h = foil.kine[1](foil.f, flow.n * flow.Δt)
        θ = foil.kine[2](foil.f, flow.n * flow.Δt, foil.ψ)
        pos = rotate_about(foil, θ + foil.θ)
        hframe = rotation(foil.θ) * [0 h]'
        pos .+= hframe        
    else
        pos = ([foil._foil[1, :] .- foil.pivot foil._foil[2, :] .+
        foil.kine.(foil._foil[1, :],foil.f,foil.k,flow.n * flow.Δt,foil.ψ)]
         * rotation(-foil.θ))'
    end
    pos .+= LE
end

function move_foil!(foil::Foil, pos)
    foil.foil = pos
    norms!(foil)
    set_collocation!(foil)
    move_edge!(foil, flow)
    flow.n += 1
end



function rotate_about!(foil, θ; pivot = foil.pivot)
    foil.foil = ([foil._foil[1, :] .- foil.pivot foil._foil[2, :]] * rotation(θ))'
    foil.foil[1, :] .+= pivot
    nothing
end
function rotate_about(foil, θ)
    pos = ([foil._foil[1, :] .- foil.pivot foil._foil[2, :]] * rotation(θ))'
    pos[1, :] .+= foil.pivot
    pos
end


"""
create_foils(num_foils, starting_positions, kine; kwargs...)

Create multiple foils with specified starting positions and kinematics.

# Arguments
- `num_foils`: Number of foils to create.
- `starting_positions`: Matrix of starting positions for each foil.
- `kine`: Kinematics for the foils.
- `kwargs`: Additional keyword arguments.

# Returns
- `foils`: Array of created foils.
- `flow`: Flow value.

# Example
An example of usage can be found in scripts/multipleSwimmers.jl
"""
function create_foils(num_foils, starting_positions, kine; kwargs...) 
    pos = deepcopy(defaultDict)       
    pos[:kine] = kine
    foils = Vector{Foil{pos[:T]}}(undef, num_foils)
    flow = 0
    for i in 1:num_foils

        for (k,v) in kwargs
            if size(v,1) == 1
                pos[k] = v
            else
                v = v[i,:]
                if size(v,1) == 1
                    pos[k] = v[1]
                else
                    pos[k] = v
                end
            end
        end   
        # @show pos     
        foil, flow = init_params(; pos...)      
          
        foil.foil[1, :] .+= starting_positions[1, i] * foil.chord
        foil.foil[2, :] .+= starting_positions[2, i]
        foil.LE = [minimum(foil.foil[1, :]), foil.foil[2, (foil.N ÷ 2 + 1)]]
        norms!(foil)
        set_collocation!(foil)
        move_edge!(foil, flow;startup=true)
        foil.edge[1, end] = 2.0 * foil.edge[1, 2] - foil.edge[1, 1]
        foils[i] =  foil
    end
    foils, flow
end
