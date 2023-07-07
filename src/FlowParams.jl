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
	naca0012 = make_naca(N + 1; chord = chord, thick=T(kwargs[:thick])) .|> T
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
	foil = Foil{T}(kine, T(f), T(kwargs[:k]), N, naca0012, copy(naca0012), col, zeros(T, N), zeros(T, N), edge, zeros(T, 2), 1, nxny, txty, ll[:], zeros(size(nxny)),zeros(size(nxny)),zeros(T, 3,2),zeros(T,2), T(kwargs[:pivot]))
    foil, fp
end