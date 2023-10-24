#  FlowParams struct and related functions

mutable struct FlowParams{T}
    Δt::T #time-step
    Uinf::T
    ρ::T #fluid density
    N::Int #number of time steps
    Ncycles::Int #number of cycles
    n::Int #current time step
    δ::T #desing radius
end

"""
    init_params(;N=128, chord = 1.0, T=Float32 )

TODO Add wake stuff
"""
function init_params(; kwargs...)
    T = kwargs[:T]
    N = kwargs[:N]
    chord = T(kwargs[:chord])
    # foil = make_naca(N + 1; chord = chord, thick=T(kwargs[:thick])) .|> T
    foil = eval(kwargs[:foil_type])(N + 1; chord = chord, thick = T(kwargs[:thick])) .|> T
    if haskey(kwargs, :motion_parameters)
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

    txty, nxny, ll = norms(foil)

    col = (get_mdpts(foil) .+ repeat(0.001 .* ll, 2, 1) .* -nxny) .|> T

    edge_vec = Uinf * Δt * [(txty[1, end] - txty[1, 1]), (txty[2, end] - txty[2, 1])] .|> T
    edge = [foil[:, end] (foil[:, end] .+ 0.5 * edge_vec) (foil[:, end] .+ 1.5 * edge_vec)]
    #  kine, f,    k,             N, _foil, foil ,     col,  σs,          
    foil = Foil{T}(kine,
        T(f),
        T(kwargs[:k]),
        N,
        foil,
        copy(foil),
        col,
        zeros(T, N),
        # μs,          edge, μ_edge,     chord, normals, tangents, panel_lengths,
        zeros(T, N),
        edge,
        zeros(T, 2),
        T(1),
        nxny,
        txty,
        ll[:],
        #   pan_vel,         wake_ind_vel,      ledge,         μ_ledge,    pivot,
        zeros(size(nxny)),
        zeros(size(nxny)),
        zeros(T, 3, 2),
        zeros(T, 2),
        T(kwargs[:pivot]),
        #   LEpos,    θ,  body vel
        zeros(T, 2),
        T(0.0),
        zeros(T, 2))
    foil, fp
end
