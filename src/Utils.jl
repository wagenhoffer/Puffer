#  Utility functions and imports
using LinearAlgebra
using Plots
using BenchmarkTools
using StaticArrays
using ForwardDiff
using BSplineKit
using Statistics
using SpecialFunctions

defaultDict = Dict(:T => Float64,
    :N => 64,
    :foil_type => :make_naca,
    :kine => :make_ang,
    :f => 1,
    :k => 1,
    :ψ => -π / 2.0,
    :chord => 1.0,
    :Nt => 150,
    :Ncycles => 1,
    :Uinf => 1.0,
    :ρ => 1000.0,
    :Nfoils => 1,
    :pivot => 0.0,
    :thick => 0.12)

"""
Plotting Recipes
"""

@recipe function f(foil::Foil)
    aspect_ratio := true
    (foil.col[1, :], foil.col[2, :])
end

@recipe function f(foils::Vector{Foil{T}}) where {T}
    aspect_ratio := true
    for foil in foils
        @series begin
            seriestype := :path
            label := ""
            (foil.col[1, :], foil.col[2, :])
        end        
    end
end

@recipe function f(foil::Foil, wake::Wake)
    # max_val = maximum(abs, wake.Γ)
    max_val = std(wake.Γ) / 2.0
    aspect_ratio := true
    @series begin
        seriestype := :path
        label := ""
        (foil.col[1, :], foil.col[2, :])
    end
    @series begin 
        seriestype := :scatter
        markersize := 3
        msw := 0
        marker_z := -wake.Γ
        color := cgrad(:coolwarm)
        clim := (-max_val, max_val)
        label := ""
        (wake.xy[1, :], wake.xy[2, :])
    end
end

@recipe function f(foils::Vector{Foil{T}}, wake::Wake) where {T}        
    aspect_ratio := true
    for foil in foils
        @series begin
            seriestype := :path
            label := ""
            (foil.col[1, :], foil.col[2, :])
        end
    end
    max_val = std(wake.Γ) / 2.0
    @series begin 
        seriestype := :scatter
        markersize := 3
        msw := 0
        marker_z := -wake.Γ
        color := cgrad(:coolwarm)
        clim := (-max_val, max_val)
        label := ""
        (wake.xy[1, :], wake.xy[2, :])
    end
end

function plot_with_normals(foil::Foil)
    plot(foil.foil[1, :], foil.foil[2, :], aspect_ratio = :equal, label = "")
    quiver!(foil.col[1, :],
        foil.col[2, :],
        quiver = (foil.normals[1, :], foil.normals[2, :]))
end

"""
    plot_coeffs(coeffs, flow)

Plot the given coefficients over the given flow parameters.
"""
function plot_coeffs(coeffs, flow)
    t = range(0, stop = flow.Ncycles * flow.N * flow.Δt, length = flow.Ncycles * flow.N)
    start = flow.N
    a = plot(t[start:end], coeffs[1, start:end], label = "Force", lw = 3, marker = :circle)
    b = plot(t[start:end], coeffs[2, start:end], label = "Lift", lw = 3, marker = :circle)
    c = plot(t[start:end], coeffs[3, start:end], label = "Thrust", lw = 3, marker = :circle)
    d = plot(t[start:end], coeffs[4, start:end], label = "Power", lw = 3, marker = :circle)
    p = plot(a, b, c, d, layout = (2, 2), legend = :topleft, size = (800, 800))
    p
end
function plot_coeffs_many(coeffs, flow)
    ps = []
    for i in 1:size(coeffs,1)
        push!(ps, plot_coeffs(coeffs[i,:, :], flow))
    end
    plot(ps..., size = (800, 800))
end


"""
    cycle_averaged(coeffs::Array{Real, 2}, flow::FlowParams, skip_cycles::Int64 = 0, sub_cycles::Int64 = 1)

Calculate the averaged values of the given coefficients over the cycles defined by the given flow parameters.
It can optionally skip a number of initial cycles and further divide each cycle into smaller sub-cycles for more detailed averaging.

# Arguments
- `coeffs`: A 2D array of coefficients to be averaged.
- `flow`: An instance of `FlowParams` struct that defines the flow parameters including the total number of cycles (`Ncycles`) and the number of data points in each cycle (`N`).
- `skip_cycles` (optional, default `0`): The number of initial cycles to skip before beginning the averaging.
- `sub_cycles` (optional, default `1`): The number of sub-cycles into which each cycle is further divided for the averaging process.

# Returns
- `avg_vals`: A 2D array with the averaged values over each (sub-)cycle.
"""
function cycle_averaged(coeffs, flow::FlowParams, skip_cycles = 0, sub_cycles = 1)
    # Assertions for error handling
    @assert skip_cycles>=0 "skip_cycles should be a non-negative integer"
    @assert sub_cycles>0 "sub_cycles should be a positive integer"

    avg_vals = zeros(4, (flow.Ncycles - skip_cycles) * sub_cycles)
    for i in (skip_cycles * sub_cycles + 1):(flow.Ncycles * sub_cycles)
        avg_vals[:, i - skip_cycles * sub_cycles] = sum(coeffs[:,
                ((i - 1) * flow.N ÷ sub_cycles + 1):((i) * flow.N ÷ sub_cycles)],
            dims = 2)
    end
    avg_vals ./= flow.N / sub_cycles
end

# # Test cases
# let
#     flow = FlowParams(10, 100) # 10 cycles, each with 100 data points
#     coeffs = rand(4, flow.Ncycles * flow.N) # Random coefficients

#     avg_vals = cycle_averaged(coeffs, flow)
#     @test size(avg_vals) == (4, 10)

#     avg_vals = cycle_averaged(coeffs, flow, 2)
#     @test size(avg_vals) == (4, 8)

#     avg_vals = cycle_averaged(coeffs, flow, 0, 2)
#     @test size(avg_vals) == (4, 20)
# end

"""
    spalarts_prune!(wake::Wake, flow::FlowParams, foil::Foil; keep=0)

Perform a modification Spalart's wake pruning procedure for a vortex method simulation.
Spalart, P. R. (1988). Vortex methods for separated flows.
# Arguments
- `wake`: A Wake object representing the vortex wake.
- `flow`: A FlowParams object representing the flow parameters.
- `foil`: A Foil object representing the airfoil.
- `keep`: An optional argument (defaulting to 0) that determines how many of the most recent vortices
          are retained without pruning.

This function modifies the `wake` object in-place. It prunes the vortices by aggregating pairs of vortices 
that satisfy Spalart's criterion and have the same sign of circulation, indicating they were generated 
during the same phase of motion. The function updates the positions, circulation, and velocity fields of
the remaining vortices in the wake.

"""

function spalarts_prune!(wake::Wake, flow::FlowParams, foil::Foil; keep = 0)
    # magic numbers from Spalart paper
    V0 = 10e-4 * flow.Uinf
    D0 = 0.1 * foil.chord
    te = foil.foil[:, 1]
    ds = sqrt.(sum(abs2, wake.xy .- te, dims = 1))
    zs = sqrt.(sum(wake.xy .^ 2, dims = 1))

    mask = abs.(wake.Γ * wake.Γ') .* abs.(zs .- zs') ./
           (abs.(wake.Γ .+ wake.Γ') .* (D0 .+ ds) .^ 1.5 .* (D0 .+ ds') .^ 1.5) .< V0

    k = 2
    num_vorts = length(wake.Γ)
    # the k here in what to keep
    # to save the last half of a cycle of motion keep = flow.N ÷ 2    
    while k < num_vorts - keep
        j = k + 1
        while j < foil.N + 1
            if mask[k, j]
                # only aggregate vortices of similar rotation
                if sign(wake.Γ[k]) == sign(wake.Γ[j])
                    wake.xy[:, j] = (abs(wake.Γ[j]) .* wake.xy[:, j] +
                                     abs(wake.Γ[k]) .* wake.xy[:, k]) ./
                                    (abs(wake.Γ[j] + wake.Γ[k]))
                    wake.Γ[j] += wake.Γ[k]
                    wake.xy[:, k] = [0.0, 0.0]
                    wake.Γ[k] = 0.0
                    mask[k, :] .= 0
                else
                    k += 1
                end
            end
            j += 1
        end
        k += 1
    end

    # clean up the wake struct
    keepers = findall(x -> x != 0.0, wake.Γ)
    wake.Γ = wake.Γ[keepers]
    wake.xy = wake.xy[:, keepers]
    wake.uv = wake.uv[:, keepers]
    nothing
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
An example of usage can be found in multipleSwimmers.jl
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

function school_of_four(height, width, chord)
    # Define the coordinates of the swimmers in the rhombus 
    xs = [0, width/2, width, width/2] .- chord/2.0
    ys = [height/2, 0, height/2, height]
    [xs ys]'
end