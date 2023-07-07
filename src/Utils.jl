#  Utility functions and imports
using LinearAlgebra
using Plots
using BenchmarkTools
using StaticArrays
using ForwardDiff
using BSplineKit
using Statistics
using SpecialFunctions

defaultDict = Dict(:T     => Float64,
	:N     => 64,
	:kine  => :make_ang,
	:f     => 1,
	:k     => 1,
	:chord => 1.0,
	:Nt    => 150,
    :Ncycles => 1,
	:Uinf   => 1.0,
	:ρ      => 1000.0, 
    :Nfoils => 1,
    :pivot  => 0.0, 
    :thick => 0.12)

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
function plot_current(foils::Vector{Foil{T}}, wake::Wake; window=nothing) where T <: Real
    if !isnothing(window)
        xs = (window[1], window[2])
    else
        xs = :auto
    end
    max_val = maximum(abs, wake.Γ)
    max_val = std(wake.Γ) / 2.0
    a = plot()
    for foil in foils
        plot!(a,foil.foil[1, :], foil.foil[2, :], aspect_ratio=:equal, label="")
        plot!(a, foil.edge[1, :], foil.edge[2, :], label="")
    end
    plot!(a, wake.xy[1, :], wake.xy[2, :],    
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