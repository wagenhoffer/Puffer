using CUDA
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

""" initialization functions """
Wake() = Wake([0, 0], 0.0, [0, 0])

#initial with the buffer panel to be cancelled
function Wake(foil::Foil{T}) where {T <: Real}
    Wake{T}(reshape(foil.edge[:, end], (2, 1)), [-foil.μ_edge[end]], [0.0 0.0]')
end

function Wake(foils::Vector{Foil{T}}) where {T <: Real}
    N = length(foils)
    xy = zeros(2, N)
    Γs = zeros(N)
    uv = zeros(2, N)
    for (i, foil) in enumerate(foils)
        xy[:, i] = foil.edge[:, end]
    end
    Wake{T}(xy, Γs, uv)
end

function move_wake!(wake, flow, foils; mask = nothing ) 
    if isnothing(mask)
       move_wake!(wake, flow)
    else
        wake.xy = sdf_fence(wake, foils, flow; mask = mask)    
    end
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

"""
    body_to_wake!(wake :: Wake, foil :: Foil)

Influence of body onto the wake and the edge onto the wake
"""
function body_to_wake!(wake::Wake, foil::Foil, flow::FlowParams)
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

# if CUDA.functional()
#     vortex_to_target(sources, targets, Γs, flow) where {T <: Real} = cast_and_pull(sources, targets, Γs,flow)
# else?
vortex_to_target(sources, targets, Γs, flow)  = cpu_vortex_to_target(sources, targets, Γs, flow)
# end


function cpu_vortex_to_target(sources::Matrix{T}, targets, Γs, flow) where {T <: Real}
    ns = size(sources)[2]
    nt = size(targets)[2]
    vels = zeros(T, (2, nt))
    vel = zeros(T, nt)
    for i in 1:ns
        dx = targets[1, :] .- sources[1, i]
        dy = targets[2, :] .- sources[2, i]
        @. vel = Γs[i] / (2π * sqrt((dx^2 + dy^2)^2 + flow.δ^4))
        @. vels[1, :] += dy * vel
        @. vels[2, :] -= dx * vel
    end
    vels
end

vortex_to_target(wake::Wake,flow) = vortex_to_target(wake.xy, wake.xy, wake.Γ, flow)
@inbounds @views macro dx(s,t) esc(:( ($t[1 ,:] .- $s[1,:]') )) end
@inbounds @views macro dy(s,t) esc(:( ($t[2 ,:] .- $s[2,:]') )) end
function cast_and_pull(sources, targets, Γs)
    @inbounds @views function vt!(vel,S,T,Γs,δ)
        n = size(S,2)
        m = size(T,2)
        mat = CUDA.zeros(Float32, m,n)    
        S = S|>CuArray
        T = T|>CuArray
        Γs = Γs|>CuArray
        dx = @dx(S,T)
        dy = @dy(S,T)
        @. mat = Γs' /(2π * sqrt((dx.^2 .+ dy.^2 )^2 + δ^4))
        @views vel[1,:] = sum(dy .* mat, dims = 2)
        @views vel[2,:] = -sum(dx .* mat, dims = 2)
        return nothing
    end
    vel = CUDA.zeros(Float32, size(targets)...)
    vt!(vel, sources, targets, Γs, flow.δ)
    vel|>Array    
end

function release_vortex!(wake::Wake, foil::Foil)    
    wake.xy = [wake.xy foil.edge[:, 2]]
    wake.Γ = [wake.Γ..., (foil.μ_edge[1] - foil.μ_edge[2])]
    # Set all back to zero for the next time step
    wake.uv = [wake.uv .* 0.0 [0.0, 0.0]]

    if any(foil.μ_ledge .!= 0)
        wake.xy = [wake.xy foil.ledge[:, 3]]
        wake.Γ = [wake.Γ..., (foil.μ_ledge[1] - foil.μ_ledge[2])]
        wake.uv = [wake.uv .* 0.0 [0.0, 0.0]]
    end
    nothing
end

"""
    add_vortex!(wake::Wake, foil::Foil)

Reduced order model for adding a vortex to the wake
"""
function add_vortex!(wake::Wake, foil::Foil, Γ, pos = nothing)
    if isnothing(pos)
        pos = (foil.col[:, 1] + foil.col[:, end])/2.0
    end        
    wake.xy = [wake.xy pos]
    wake.Γ = [wake.Γ..., Γ]
    # Set all back to zero for the next time step
    wake.uv = [wake.uv .* 0.0 [0.0, 0.0]]
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

function cancel_buffer_Γ!(wake::Wake, foils::Vector{Foil{T}}) where {T <: Real}
    for (i, foil) in enumerate(foils)
        wake.xy[:, i] = foil.edge[:, 2]
        wake.Γ[i] = -foil.μ_edge[end]
    end
end
